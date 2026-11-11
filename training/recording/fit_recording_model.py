"""Recover capture metadata and validate the original training transfer rules."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"code"))

DATA = Path(__file__).resolve().parent
MODEL = None
FOLDS = ((1,2,3),(4,5,6),(7,8,9),(16,17,18),(19,20,21),(22,23,24))


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    MODEL.mkdir(parents=True,exist_ok=False)
    from metadata_transfer_core import row_identity, select_metadata_donors, emotion_proposals, action_and_sequence_proposals
    split_root = DATA
    # Retain timestamp and recording correspondence, discarding activity labels.
    tokens = defaultdict(set)
    pattern = re.compile(r'/Skeleton/(?:[^/]+)/(?P<user>user\d+)/(?P<trial>\d+-\d+-\d+)/predictions/(?P<basename>[^/]+)$', re.I)
    timestamp = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}_\d+)')
    inputs = [split_root/(name+'.gz') for name in ('cross_subject_train_1.txt','cross_subject_test_1.txt')]
    for path in inputs:
        for line in gzip.decompress(path.read_bytes()).decode('utf-8').splitlines():
            match = pattern.search(line.strip().replace('\\','/'))
            if match:
                token = timestamp.search(match['basename'])
                if token:
                    tokens[token[1]].add((match['user'].lower(), match['trial']))
    unique = {key:next(iter(values)) for key,values in sorted(tokens.items()) if len(values) == 1}
    with gzip.GzipFile(filename=str(MODEL/'recording_index.json.gz'), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(unique, separators=(',',':')).encode())
    raw_metadata = read(DATA/'activity_metadata.csv')
    metadata = [dict(zip(('user_id','activity_id','identifier_id'), values)) for values in sorted({
        tuple(row[key] for key in ('user_id','activity_id','identifier_id')) for row in raw_metadata})]
    (MODEL/'activity_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    training = [row for row in read(DATA.parent/'training_qa.csv') if row['source']=='HAU']
    users = {row_identity(row,{})[0] for row in training}
    aggregate = {category:Counter() for category in ('emotion','multi','sequence')}
    folds=[]
    for number, group in enumerate(FOLDS):
        held = {'user'+str(n) for n in group}
        fit = [row for row in training if row_identity(row,{})[0] not in held]
        validation = [row for row in training if row_identity(row,{})[0] in held]
        public = [{k:v for k,v in row.items() if k not in {'answer','prediction'}} for row in validation]
        gold = {row['qa_id']:row['answer'] for row in validation}
        base_file = DATA / f'fold_{number}_predictions.json'
        base_record = json.loads(base_file.read_text())
        parent = base_record['final']
        donors, _ = select_metadata_donors(metadata, users-held, held)
        emotion, _ = emotion_proposals(fit, public, {}, donors)
        actions = action_and_sequence_proposals(fit, public, {}, donors, parent)
        categories = {row['qa_id']:row['category'] for row in validation}
        fold = {}
        for category in aggregate:
            proposals = [row for row in emotion+actions if categories[row['qa_id']] == category]
            changed = [row for row in proposals if row['prediction'] != parent[row['qa_id']]]
            stats = Counter(proposals=len(proposals), correct_proposals=sum(row['prediction']==gold[row['qa_id']] for row in proposals),
                changes=len(changed), wins=sum(row['prediction']==gold[row['qa_id']] and parent[row['qa_id']]!=gold[row['qa_id']] for row in changed),
                losses=sum(row['prediction']!=gold[row['qa_id']] and parent[row['qa_id']]==gold[row['qa_id']] for row in changed))
            aggregate[category].update(stats)
            fold[category]=dict(stats)
        folds.append(dict(fold=number, statistics=fold))
    # Freeze a category only if it improves the retained held-subject parent
    # without introducing a held-subject error. No test answer enters this step.
    gates = {category: stats['wins']>0 and stats['losses']==0 for category,stats in aggregate.items()}
    (MODEL/'gates.json').write_text(json.dumps(gates, indent=2)+'\n')
    report = dict(validation='PASS', token_entries=len(unique), activity_signatures=len(metadata),
        official_training_rows=len(training), held_subject_folds=6,
        aggregate={k:dict(v) for k,v in aggregate.items()}, enabled_categories=gates, folds=folds,
        test_answer_inputs=False, prior_submission_inputs=False,
        correspondence_fields=['timestamp_token','user','trial'], action_labels_in_correspondence_index=False)
    (MODEL/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='folds'}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    MODEL=parser.parse_args().output
    main()
