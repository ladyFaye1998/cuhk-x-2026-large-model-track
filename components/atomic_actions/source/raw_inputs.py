from __future__ import annotations
import csv,hashlib,json,re,struct,time,zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path,PurePosixPath
import numpy as np
import requests
TEST_ARCHIVE_ID='1seN5q84NJ8IQQBE6c2X1oE9UQNBpLiK4'
TEST_ARCHIVE_SIZE=2_786_949_673
POOL=128
N_JOINTS=17
RANGE_WORKERS=8
def array_sha256(value,dtype):
    return hashlib.sha256(np.asarray(value,dtype=dtype,order='C').tobytes()).hexdigest()

def direct_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def fetch_range(session: requests.Session, start: int, end: int, attempts: int = 8) -> bytes:
    expected = end - start + 1
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(
                direct_url(TEST_ARCHIVE_ID),
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"},
                timeout=(30, 240),
            )
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != expected:
                raise RuntimeError(f"range status={response.status_code} got={len(response.content)} expected={expected}")
            return response.content
        except Exception as error:
            last_error = error
            time.sleep(min(30.0, 1.8 ** attempt))
    raise RuntimeError(f"range {start}-{end} failed: {last_error}")


def parse_zip64_extra(extra: bytes, uncomp_size: int, comp_size: int, local_offset: int, disk_start: int):
    pos = 0
    while pos + 4 <= len(extra):
        field_id, field_len = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4:pos + 4 + field_len]
        pos += 4 + field_len
        if field_id != 0x0001:
            continue
        cursor = 0
        def take(fmt: str):
            nonlocal cursor
            size = struct.calcsize(fmt)
            if cursor + size > len(body):
                raise RuntimeError("truncated ZIP64 extra field")
            value = struct.unpack_from(fmt, body, cursor)[0]
            cursor += size
            return value
        if uncomp_size == 0xFFFFFFFF:
            uncomp_size = take("<Q")
        if comp_size == 0xFFFFFFFF:
            comp_size = take("<Q")
        if local_offset == 0xFFFFFFFF:
            local_offset = take("<Q")
        if disk_start == 0xFFFF:
            disk_start = take("<L")
        break
    return int(uncomp_size), int(comp_size), int(local_offset), int(disk_start)


def parse_entries(data: bytes):
    pos = 0
    while pos + 46 <= len(data):
        if data[pos:pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"central-directory signature missing at {pos}")
        values = struct.unpack_from("<4s6H3L5H2L", data, pos)
        flags, method, crc = values[3], values[4], values[7]
        comp_size, uncomp_size = values[8], values[9]
        name_len, extra_len, comment_len = values[10:13]
        disk_start, local_offset = values[13], values[16]
        name_start = pos + 46
        name_bytes = data[name_start:name_start + name_len]
        extra = data[name_start + name_len:name_start + name_len + extra_len]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437", errors="replace")
        uncomp_size, comp_size, local_offset, disk_start = parse_zip64_extra(
            extra, uncomp_size, comp_size, local_offset, disk_start
        )
        yield {
            "name": name, "disk": disk_start, "offset": local_offset,
            "compressed": comp_size, "uncompressed": uncomp_size,
            "method": int(method), "flags": int(flags), "crc": int(crc),
        }
        pos += 46 + name_len + extra_len + comment_len
    if pos != len(data):
        raise RuntimeError(f"unparsed central-directory bytes: {len(data) - pos}")


def central_directory_info(session: requests.Session) -> dict[str, int]:
    tail_size = 4 * 1024 * 1024
    tail_start = TEST_ARCHIVE_SIZE - tail_size
    tail = fetch_range(session, tail_start, TEST_ARCHIVE_SIZE - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("EOCD not found")
    values = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
    info = {"entries": int(values[4]), "size": int(values[5]), "offset": int(values[6])}
    if info["entries"] == 0xFFFF or info["size"] == 0xFFFFFFFF or info["offset"] == 0xFFFFFFFF:
        locator_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
        if locator_pos < 0:
            raise RuntimeError("ZIP64 locator not found")
        locator = struct.unpack_from("<4sLQL", tail, locator_pos)
        record = fetch_range(session, int(locator[2]), int(locator[2]) + 95)
        if record[:4] != b"PK\x06\x06":
            raise RuntimeError("ZIP64 EOCD not found")
        zip64 = struct.unpack_from("<4sQ2H2L4Q", record, 0)
        info = {"entries": int(zip64[7]), "size": int(zip64[8]), "offset": int(zip64[9])}
    return info


def is_skeleton(entry: dict) -> bool:
    parts = PurePosixPath(entry["name"]).parts
    return "Skeleton" in parts and entry["name"].endswith(".json")


def merged_intervals(entries: list[dict]):
    output = []
    for entry in sorted(entries, key=lambda row: row["offset"]):
        start = int(entry["offset"])
        end = start + int(entry["compressed"]) + 1024 - 1
        if not output or start > output[-1][1] + 1:
            output.append([start, end, [entry]])
        else:
            output[-1][1] = max(output[-1][1], end)
            output[-1][2].append(entry)
    return output


class SparseStore:
    def __init__(self, session: requests.Session, entries: list[dict]):
        self.payloads: dict[str, bytes] = {}
        intervals = merged_intervals(entries)
        print(f"Downloading {len(entries)} Skeleton JSON files in {len(intervals)} archive ranges", flush=True)
        def load(item):
            start, end, rows = item
            return start, fetch_range(session, int(start), int(end)), rows
        with ThreadPoolExecutor(max_workers=RANGE_WORKERS) as pool:
            futures = [pool.submit(load, item) for item in intervals]
            for number, future in enumerate(as_completed(futures), 1):
                start, blob, rows = future.result()
                for entry in rows:
                    relative = int(entry["offset"]) - int(start)
                    header = blob[relative:relative + 30]
                    if header[:4] != b"PK\x03\x04":
                        raise RuntimeError(f"bad local header for {entry['name']}")
                    values = struct.unpack("<4s5H3L2H", header)
                    data_start = relative + 30 + int(values[9]) + int(values[10])
                    payload = blob[data_start:data_start + int(entry["compressed"])]
                    if int(entry["method"]) == 8:
                        payload = zlib.decompress(payload, -15)
                    elif int(entry["method"]) != 0:
                        raise RuntimeError(f"unsupported ZIP method {entry['method']}")
                    if len(payload) != int(entry["uncompressed"]) or (zlib.crc32(payload) & 0xFFFFFFFF) != int(entry["crc"]):
                        raise RuntimeError(f"ZIP integrity mismatch for {entry['name']}")
                    self.payloads[str(entry["name"])] = payload
                if number % 100 == 0 or number == len(intervals):
                    print(f"Archive ranges {number}/{len(intervals)}", flush=True)
    def raw(self, entry: dict) -> bytes:
        return self.payloads[str(entry["name"])]
    def close(self) -> None:
        self.payloads.clear()

def numeric_key(value: str):
    numbers = tuple(int(item) for item in re.findall(r"\d+", str(value)))
    return numbers or (10 ** 9,)


def test_group(name: str) -> str:
    return next(part for part in PurePosixPath(name).parts if part.startswith("SM_test_"))


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def decode_sequence(store: SparseStore, entries: list[dict]) -> tuple[np.ndarray, int]:
    frames = []
    for entry in sorted(entries, key=lambda row: numeric_key(PurePosixPath(row["name"]).name)):
        try:
            value = json.loads(decode_text(store.raw(entry)))
            people = value if isinstance(value, list) else [value]
            best, best_spread = None, -1.0
            for person in people:
                if not isinstance(person, dict):
                    continue
                points = np.asarray(person.get("keypoints", []), dtype=np.float32)
                if points.shape != (N_JOINTS, 3) or not np.isfinite(points).any():
                    continue
                points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
                spread = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
                if spread > best_spread:
                    best, best_spread = points, spread
            if best is not None:
                frames.append(best)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("clip has no valid Skeleton frames")
    values = np.stack(frames).astype(np.float32)
    if len(values) > POOL:
        selected = np.linspace(0, len(values) - 1, POOL).round().astype(np.int64)
        values = values[selected]
    output = np.zeros((POOL, N_JOINTS, 3), dtype=np.float32)
    output[:len(values)] = values
    return output, int(len(values))


def read_sample_paths(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise RuntimeError("sample submission schema")
        paths = [str(row["path"]).replace("\\", "/") for row in reader]
    if len(paths) != 405 or len(set(paths)) != 405:
        raise RuntimeError("sample submission row closure")
    return paths

def download_test(template):
    raw_start = time.perf_counter()
    session = requests.Session()
    central_info = central_directory_info(session)
    central = fetch_range(session, central_info["offset"], central_info["offset"] + central_info["size"] - 1)
    entries = [entry for entry in parse_entries(central) if is_skeleton(entry)]
    if len(entries) != 9201:
        raise RuntimeError(f"Skeleton entry count drift: {len(entries)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[test_group(entry["name"])].append(entry)
    groups = sorted(grouped)
    sample_path_file = template
    sample_paths = read_sample_paths(sample_path_file)
    sample_groups = [PurePosixPath(path.rstrip("/")).name for path in sample_paths]
    if sample_groups != groups:
        raise RuntimeError("sample rows and raw archive groups do not align exactly")

    sequences = np.zeros((len(groups), POOL, N_JOINTS, 3), dtype=np.float32)
    n_valid = np.zeros(len(groups), dtype=np.int32)
    store = SparseStore(session, entries)
    try:
        for index, group in enumerate(groups):
            sequences[index], n_valid[index] = decode_sequence(store, grouped[group])
            if (index + 1) % 100 == 0 or index + 1 == len(groups):
                print(f"Decoded clips {index + 1}/{len(groups)}", flush=True)
    finally:
        store.close()

    tensor_hash = array_sha256(sequences, "<f4")
    n_valid_hash = array_sha256(n_valid, "<i8")
    if tensor_hash != 'e98d7f205ba1a753a246cda6193834c167f7b29e819210b0d369fc21d6509800' or n_valid_hash != '83f9a121745879621bb0b57e6b846068ef06155a78eb83474e78ffa9bd3b7460':
        raise RuntimeError("decoded raw Skeleton tensor drift")

    download_seconds = time.perf_counter() - raw_start
    print({"clips": len(groups), "valid_frames": int(n_valid.sum()), "tensor_sha256": tensor_hash})

    return sequences,n_valid,sample_paths,{"download_and_decode_seconds":download_seconds,"tensor_sha256":tensor_hash,"valid_count_sha256":n_valid_hash,"valid_frames":int(n_valid.sum()),"archive_file_id":TEST_ARCHIVE_ID}
