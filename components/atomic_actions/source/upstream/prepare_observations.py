"""Run the three original observation decoders with one bounded official transport."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

SOURCE=Path(__file__).resolve().parent

def fact(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def apply_transport_counters(report, receipts):
    """Report successful strict HTTP-206 traffic from the actual transport."""
    keys=('downloaded_bytes','http_requests','official_mirror_requests')
    report['original_decoder_transport_counters']={key:report[key] for key in keys if key in report}
    report.update(downloaded_bytes=sum(row['bytes'] for row in receipts),
                  http_requests=len(receipts),
                  official_mirror_requests=sum(row['transport']=='official_hf_mirror' for row in receipts),
                  transport_counter_scope='Successful exact HTTP-206 ranges from the shared transport; HTML/retries are not counted')

def run(kind,indexes,out,template):
    index_source=SOURCE/'index_source'
    sys.path.insert(0,str(index_source))
    import official_archive_transport as transport
    def original_range(file_id,start,end,attempts=8):
        return transport.fetch_range(file_id,start,end,attempts=attempts)
    if kind=='skeleton':
        sys.path.insert(0,str(SOURCE/'observations'))
        import official_skeleton as decoder
        def skeleton_range(file_id,start,end,total_size,attempts=4):
            # The shared transport validates the full archive size internally.
            return transport.fetch_range(file_id,start,end,total_size=total_size,attempts=attempts)
        decoder.fetch_range=skeleton_range
        result=decoder.prepare_all(out,template=template)
        manifest=Path(result['report'])
    elif kind=='sensor':
        sys.path.insert(0,str(SOURCE/'sensor_prepare'))
        import prepare_sensor as preparation
        original_loader=preparation.load
        def loader(name,path):
            module=original_loader(name,path)
            if name=='sensor_transport':
                module.fetch_range=original_range
                module.receipt=transport.receipts
            return module
        preparation.load=loader
        preparation.prepare(indexes,out)
        manifest=Path(out)/'preparation_manifest.json'
    elif kind=='raw':
        sys.path.insert(0,str(SOURCE/'raw'))
        import raw_adaptive_transport
        raw_adaptive_transport.fetch_range=original_range
        raw_adaptive_transport.receipt=transport.receipts
        import raw_adaptive_prepare
        raw_adaptive_prepare.run_preparation(indexes,out,transport='official-mirror')
        manifest=Path(out)/'preparation_manifest.json'
    else:
        raise ValueError('Unknown observation decoder')
    value=json.loads(manifest.read_text())
    range_receipts=transport.receipts()
    apply_transport_counters(value,range_receipts)
    value['end_to_end_transport']={'source_files':[fact(__file__),fact(transport.__file__)],
                                  'receipt':range_receipts,
                                  'numerical_decoder_changes':False}
    value['official_index_manifest']=fact(Path(indexes)/'index_generation_report.json')
    manifest.write_text(json.dumps(value,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=['skeleton','sensor','raw'])
    p.add_argument('--indexes',required=True);p.add_argument('--out',required=True);p.add_argument('--template',required=True)
    a=p.parse_args();print(run(a.kind,a.indexes,a.out,a.template))
