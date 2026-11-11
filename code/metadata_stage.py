"""Recording correspondence and training-example transfer for Large questions."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import re
from metadata_transfer_core import (select_metadata_donors, emotion_proposals,
                                    action_and_sequence_proposals, row_identity)

TOKEN = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}_\d+)')


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identify(rows, sensors, index):
    names = {Path(r['path'].replace('\\', '/')).parts[1]
             for r in rows if r['source'] == 'HAU'}
    sensors = Path(sensors)
    if (sensors / 'large_model_track_test').is_dir():
        sensors = sensors / 'large_model_track_test'
    roots = {p.name: p for p in sensors.iterdir() if p.is_dir()}
    accepted, evidence = {}, []
    for name in sorted(names):
        folder = roots.get(name)
        if folder is None:
            continue
        votes = Counter()
        for path in (folder / 'Skeleton').rglob('*.json'):
            match = TOKEN.search(path.name)
            if match and match[1] in index:
                votes[tuple(index[match[1]])] += 1
        if len(votes) == 1 and sum(votes.values()) >= 2:
            user, trial = next(iter(votes))
            accepted[name] = dict(user=user, trial=trial)
        evidence.append(dict(recording=name, matched_frames=sum(votes.values()),
            consistent_identities=len(votes), accepted=name in accepted))
    return accepted, evidence


def apply(rows, parent, training, identities, metadata, gates):
    training_users = {row_identity(row, {})[0] for row in training if row['source'] == 'HAU'}
    target_users = {value['user'] for value in identities.values()}
    donors, donor_evidence = select_metadata_donors(metadata, training_users, target_users)
    # Restrict proposal generation to recordings that have a correspondence.
    eligible = [row for row in rows if row['source'] == 'HAU'
                and Path(row['path'].replace('\\','/')).parts[1] in identities]
    emotion, emotion_evidence = emotion_proposals(training, eligible, identities, donors)
    action = action_and_sequence_proposals(training, eligible, identities, donors, parent)
    output = dict(parent)
    changes = []
    categories = {row['qa_id']: row['category'] for row in rows}
    for proposal in emotion + action:
        key = proposal['qa_id']
        category = categories[key]
        if gates.get(category, False) and proposal['prediction'] != output[key]:
            changes.append(dict(qa_id=key, category=category, before=output[key], after=proposal['prediction']))
            output[key] = proposal['prediction']
    return output, dict(accepted_changes=changes, donors=donor_evidence,
                       emotion_availability=emotion_evidence,
                       proposal_counts=dict(Counter(categories[p['qa_id']] for p in emotion+action)))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = read(args.questions)
    if any(row.get(key, '').strip() for row in rows for key in ('answer', 'prediction', 'target', 'label')):
        raise ValueError('Questions contain answer values')
    execution = json.loads((args.base.parent / 'execution.json').read_text())
    if not execution['historical_prediction_inputs'] is False:
        raise ValueError('Parent execution contract differs')
    if execution['final_sha256'] != sha(args.base):
        raise ValueError('Parent predictions differ from execution record')
    # Parent raw stage receipts bind its computation to these questions.
    for branch in execution['raw_model_runs'].values():
        folder = Path(branch)
        receipt = folder / ('inference_manifest.json' if (folder/'inference_manifest.json').is_file() else 'execution.json')
        if json.loads(receipt.read_text())['questions_sha256'] != sha(args.questions):
            raise ValueError('Parent raw observations belong to other questions')
    parent = {row['qa_id']:row['prediction'] for row in read(args.base)}
    if set(parent) != {row['qa_id'] for row in rows}:
        raise ValueError('Parent question coverage differs')
    with gzip.open(args.model / 'recording_index.json.gz', 'rt', encoding='utf-8') as stream:
        index = json.load(stream)
    identities, alignment = identify(rows, args.sensors, index)
    metadata = json.loads((args.model / 'activity_metadata.json').read_text())
    gates = json.loads((args.model / 'gates.json').read_text())
    training = read(args.training / 'training_qa.csv')
    output, details = apply(rows, parent, training, identities, metadata, gates)
    with (args.output/'submission.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\r\n')
        writer.writerow(['qa_id','prediction'])
        writer.writerows((row['qa_id'], output[row['qa_id']]) for row in rows)
    report = dict(rows=len(rows), matched_recordings=len(identities), questions_sha256=sha(args.questions),
        output_sha256=sha(args.output/'submission.csv'), historical_submission_inputs=False,
        alignment=alignment, **details)
    (args.output / 'execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'alignment','donors','emotion_availability','output_sha256','questions_sha256','accepted_changes'}} | {'changes':len(details['accepted_changes'])}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('questions','sensors','base','model','training','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
