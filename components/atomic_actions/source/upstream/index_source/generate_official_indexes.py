"""Generate fresh nonvisual ZIP-directory indexes from official archive ranges."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import time
import zlib

import official_archive_transport
from zip_index_codec import parse_entries

ARCHIVES = {
    "train": {"file_id":"1nlJIpcRuL2FRcapJaFVS3eC-uZuedtqh", "size":1673136305, "last_disk":8},
    "test": {"file_id":"1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4", "size":2786949673, "last_disk":0},
}
NONVISUAL = {"IMU", "Radar", "Skeleton"}
EXPECTED_COUNTS = {"train":{"IMU":5806,"Radar":2914,"Skeleton":86050},
                   "test":{"IMU":810,"Radar":404,"Skeleton":9203}}
REQUIRED_FIELDS = ("name","disk","offset","compressed","uncompressed","method","flags","crc")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temp,path)


def directory_location(split, fetch_range):
    """Locate EOCD/ZIP64 using at most 65,653 metadata bytes per archive."""
    spec = ARCHIVES[split]
    size, identifier = spec["size"], spec["file_id"]
    tail_size = min(size, 65535+22)
    tail = fetch_range(identifier,size-tail_size,size-1,size)
    pos = tail.rfind(b"PK\x05\x06")
    while pos >= 0:
        if pos+22 <= len(tail):
            fields = struct.unpack_from("<4s4H2LH",tail,pos)
            if pos+22+fields[7] == len(tail):
                break
        pos = tail.rfind(b"PK\x05\x06",0,pos)
    if pos < 0:
        raise ValueError("Official ZIP end record missing or malformed")
    disk, cd_disk, count, cd_size, offset = fields[1], fields[2], fields[4], fields[5], fields[6]
    if disk != spec["last_disk"] or cd_disk != spec["last_disk"]:
        raise ValueError("Official archive central directory is on an unexpected disk")
    zip64 = count == 0xFFFF or cd_size == 0xFFFFFFFF or offset == 0xFFFFFFFF
    if zip64:
        loc = pos-20
        if loc < 0 or tail[loc:loc+4] != b"PK\x06\x07":
            raise ValueError("ZIP64 locator missing")
        locator = struct.unpack_from("<4sLQL",tail,loc)
        if locator[1] != spec["last_disk"] or locator[3] != spec["last_disk"]+1:
            raise ValueError("ZIP64 locator archive layout differs")
        begin = locator[2]
        if begin+56 > size:
            raise ValueError("ZIP64 end record lies outside archive")
        record = fetch_range(identifier,begin,min(begin+96,size)-1,size)
        fields64 = struct.unpack_from("<4sQ2H2L4Q",record)
        if fields64[0] != b"PK\x06\x06" or fields64[1] < 44 or fields64[4] != spec["last_disk"] or fields64[5] != spec["last_disk"]:
            raise ValueError("Malformed or unexpected ZIP64 end record")
        count, cd_size, offset = fields64[7], fields64[8], fields64[9]
    if not (0 <= offset < offset+cd_size <= size) or not 0 < cd_size < 100_000_000:
        raise ValueError("Unexpected official central-directory span")
    if split == "train" and (offset,cd_size,count) != (1599292780,73843427,491879):
        raise ValueError("Official training archive directory metadata changed")
    return {"file_id":identifier,"archive_size":size,"offset":int(offset),"size":int(cd_size),"entries":int(count),"zip64":zip64}


def modality(entry, split):
    parts = [part for part in entry["name"].split("/") if part]
    if split == "train":
        if "data" not in parts:
            return None
        index = parts.index("data")+1
        return parts[index] if index < len(parts) else None
    return next((part for part in parts if part in NONVISUAL | {"Depth_Color","IR","Thermal"}),None)


def selected_entries(blob, split):
    """Preserve the original central-directory order and all original row fields."""
    for row in parse_entries(blob):
        if row["name"].endswith("/") or modality(row,split) not in NONVISUAL:
            continue
        if tuple(row) != REQUIRED_FIELDS or row["flags"] & 1 or row["method"] not in (0,8):
            raise ValueError("Unexpected nonvisual row schema, encryption or compression")
        if row["disk"] not in ({2,7} if split == "train" else {0}):
            raise ValueError("Nonvisual member points at an unexpected archive disk")
        if any(row[key] < 0 for key in REQUIRED_FIELDS[1:]) or row["crc"] > 0xFFFFFFFF:
            raise ValueError("Invalid nonvisual metadata values")
        yield row


def write_index(blob, split, destination, expected_count=None):
    destination = Path(destination)
    temp = destination.with_name(destination.name+".tmp")
    plain_hash, counts, names = hashlib.sha256(), Counter(), set()
    with temp.open("wb") as output, gzip.GzipFile(filename="",mode="wb",fileobj=output,compresslevel=6,mtime=0) as packed:
        for row in selected_entries(blob,split):
            if row["name"] in names:
                raise ValueError("Duplicate official nonvisual member name")
            names.add(row["name"])
            counts[modality(row,split)] += 1
            line = (json.dumps(row,separators=(",",":"))+"\n").encode("utf-8")
            packed.write(line)
            plain_hash.update(line)
    if expected_count is not None and dict(counts) != expected_count:
        raise ValueError("Official nonvisual member counts changed")
    os.replace(temp,destination)
    return {"file":destination.name,"rows":len(names),"modality_counts":dict(counts),
            "compressed_sha256":sha256(destination),"jsonl_sha256":plain_hash.hexdigest(),
            "bytes":destination.stat().st_size,"crc_metadata_rows":len(names)}


def verify_member_payload(payload, entry):
    """Optional downstream check, called after a raw member is decompressed."""
    if len(payload) != entry["uncompressed"] or zlib.crc32(payload)&0xFFFFFFFF != entry["crc"]:
        raise ValueError("Official raw member CRC or uncompressed length mismatch")
    return payload


def generate_indexes(out_dir, sample_template_path, transport=None):
    """Return (paths, report); inputs are official archives and the official template."""
    output = Path(out_dir).resolve()
    template = Path(sample_template_path).resolve(strict=True)
    with template.open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if "path" not in (reader.fieldnames or []):
            raise ValueError("Official sample template requires its path column")
        paths = [row["path"] for row in reader]
    if paths != [f"small_model_track_test/SM_test_{i:04d}/" for i in range(1,406)]:
        raise ValueError("Official test template rows/order differ")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty index output directory; existing evidence is preserved")
    output.mkdir(parents=True,exist_ok=True)
    transport = official_archive_transport if transport is None else transport
    reader = transport.fetch_range if hasattr(transport,"fetch_range") else transport
    range_receipts = []
    def read(identifier,start,end,size):
        blob = reader(identifier,start,end,size)
        if len(blob) != end-start+1:
            raise ValueError("Transport returned a different index metadata range")
        range_receipts.append({"file_id":identifier,"start":start,"end":end,"bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest()})
        return blob
    reports = {}
    started = time.perf_counter()
    for split in ("train","test"):
        directory = directory_location(split,read)
        print(json.dumps({"stage":"fresh_official_index","split":split,"central_directory_bytes":directory["size"]}),flush=True)
        begin,end = directory["offset"],directory["offset"]+directory["size"]
        blob = b"".join(read(directory["file_id"],p,min(p+4*1024*1024,end)-1,directory["archive_size"]) for p in range(begin,end,4*1024*1024))
        if sum(1 for _ in parse_entries(blob)) != directory["entries"]:
            raise ValueError("Parsed central-directory count differs from EOCD")
        filename = "nonvisual_entries.jsonl.gz" if split == "train" else "test_nonvisual_entries.jsonl.gz"
        reports[split] = {**write_index(blob,split,output/filename,EXPECTED_COUNTS[split]),
                          "directory":directory,"central_directory_sha256":hashlib.sha256(blob).hexdigest()}
        del blob
    shutil.copyfile(template,output/"sample_submission.csv")
    source_files = [Path(__file__),Path(__file__).with_name("zip_index_codec.py")]
    if hasattr(transport,"__file__"):
        source_files.append(Path(transport.__file__))
    report = {"status":"completed","generated_from":"fresh official ZIP central-directory ranges",
              "recovered_index_inputs_used":False,"learned_inputs_used":[],"indexes":reports,
              "sample_template_sha256":sha256(template),"source_files":[{"name":p.name,"sha256":sha256(p)} for p in source_files],
              "downloaded_bytes":sum(r["bytes"] for r in range_receipts),"range_receipts":range_receipts,
              "crc_coverage":{"directory_crc_field_preserved_for_all_selected_members":True,
                              "raw_member_payload_crc_verified_by_index_generation":False,
                              "reason":"This stage reads ZIP directories only; verify_member_payload is provided for downstream readers after decompression."},
              "gzip_encoding":"original compact JSON row order; gzip mtime=0 and empty filename, so compressed bytes differ from historical timestamped gzip",
              "elapsed_seconds":time.perf_counter()-started}
    write_json(output/"index_generation_report.json",report)
    return {"train_index":output/"nonvisual_entries.jsonl.gz","test_index":output/"test_nonvisual_entries.jsonl.gz","template":output/"sample_submission.csv"},report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--sample-template",required=True)
    args = parser.parse_args()
    paths,report = generate_indexes(args.out,args.sample_template)
    print(json.dumps({"paths":{key:str(value) for key,value in paths.items()},"downloaded_bytes":report["downloaded_bytes"],"rows":{key:value["rows"] for key,value in report["indexes"].items()}},indent=2))
