"""Verify technical package integrity, source syntax and inference input closure."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import csv
import hashlib
import io
import json
import zipfile


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or any(x in {'..', '.'} for x in path.parts) or ':' in name:
        raise ValueError('Invalid archive member path')
    return name


def verify(package):
    package = Path(package).resolve(strict=True)
    archive = None
    if package.is_dir():
        names = {str(p.relative_to(package)).replace('\\', '/') for p in package.rglob('*') if p.is_file()}
        def open_file(name):
            safe_name(name)
            resolved = (package / name).resolve(strict=True)
            if not resolved.is_relative_to(package):
                raise ValueError('Package member escapes root')
            return resolved.open('rb')
    else:
        archive = zipfile.ZipFile(package)
        try:
            members = [i for i in archive.infolist() if not i.is_dir()]
            raw_names = [safe_name(i.filename) for i in members]
            if len(set(raw_names)) != len(raw_names):
                raise ValueError('Duplicate archive member')
            roots = [n for n in raw_names if n == 'MANIFEST.json' or n.count('/') == 1 and n.endswith('/MANIFEST.json')]
            if len(roots) != 1:
                raise ValueError('Expected one package root manifest')
            prefix = roots[0][:-len('MANIFEST.json')]
            if any(not name.startswith(prefix) for name in raw_names):
                raise ValueError('Archive member outside package root')
            names = {n[len(prefix):] for n in raw_names}
            def open_file(name):
                safe_name(name)
                return archive.open(prefix + name)
        except Exception:
            archive.close()
            raise
    try:
        def read(name):
            with open_file(name) as stream:
                return stream.read()
        manifest = json.loads(read('MANIFEST.json'))
        expected = manifest['files']
        for name in expected:
            safe_name(name)
        if names != set(expected) | {'MANIFEST.json'}:
            raise ValueError('Package members differ from manifest')
        required = {'README.md', 'TECHNICAL_REPORT.md', 'inference.sh',
                    'code/run.py', 'code/winning_head.py', 'training/source/train_winner_head.py',
                    'checkpoints/decisions/winner_head.joblib',
                    'checkpoints/decisions/winner_head_validation.json',
                    'final_submission.csv', 'notebook/CUHKX_large_inference.ipynb',
                    'inference_asset_manifest.json', 'environment/requirements-runtime.txt'}
        if not required <= names:
            raise ValueError('Required technical package files are absent')
        if any('honor_declaration' in name.lower() for name in names):
            raise ValueError('Participant declaration belongs in the separate email attachments')
        parsed = 0
        for name, record in expected.items():
            count, h = 0, hashlib.sha256()
            with open_file(name) as stream:
                while chunk := stream.read(4 << 20):
                    count += len(chunk); h.update(chunk)
            if count != record['bytes'] or h.hexdigest() != record['sha256']:
                raise ValueError('File integrity failed: ' + name)
            if name.endswith('.py'):
                ast.parse(read(name).decode('utf-8-sig'), filename=name)
                parsed += 1
        assets = json.loads(read('inference_asset_manifest.json'))
        if assets['prediction_csv_inputs']:
            raise ValueError('Prediction CSV declared as an inference input')
        asset_names = set()
        for item in assets['files']:
            name = safe_name(item['path'])
            if name in asset_names:
                raise ValueError('Repeated inference asset')
            asset_names.add(name)
            if name not in expected or any(item[key] != expected[name][key] for key in ('bytes', 'sha256')):
                raise ValueError('Inference asset differs from package: ' + name)
            if name.endswith('.csv') and name != 'training/training_qa.csv':
                raise ValueError('Unexpected CSV in model inputs: ' + name)
        notebook = json.loads(read('notebook/CUHKX_large_inference.ipynb'))
        code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        for index, cell in enumerate(code_cells):
            source = ''.join(cell['source'])
            ast.parse(source, filename='notebook cell ' + str(index + 1))
        rows = list(csv.DictReader(io.StringIO(read('final_submission.csv').decode('utf-8-sig'))))
        if len(rows) != manifest['rows'] or len({r['qa_id'] for r in rows}) != len(rows):
            raise ValueError('Submission row count or identifier uniqueness failed')
        if any(set(r) != {'qa_id','prediction'} or not r['prediction'] or not set(r['prediction']) <= set('ABCD') or len(set(r['prediction'])) != len(r['prediction']) for r in rows):
            raise ValueError('Submission structure failed')
        head_validation = json.loads(read('checkpoints/decisions/winner_head_validation.json'))
        if head_validation.get('technical_verification') != 'PASS' or head_validation.get('fit_matches') != manifest['rows']:
            raise ValueError('Question-aware head validation failed')
        if head_validation.get('identifier_features') or head_validation.get('path_features'):
            raise ValueError('Question-aware head uses identifier or path features')
        head_source = read('code/winning_head.py').decode('utf-8-sig')
        if 'test_' in head_source or 'final_submission.csv' in head_source:
            raise ValueError('Question-aware inference source contains a fixed target selector')
        return {'technical_verification': 'PASS', 'file_integrity': 'PASS', 'source_syntax': 'PASS',
                'inference_asset_closure': 'PASS', 'submission_structure': 'PASS',
                'question_aware_head': 'PASS',
                'files': len(expected), 'python_sources': parsed, 'notebook_code_cells': len(code_cells), 'submission_rows': len(rows),
                'verification_scope': 'Archive, source syntax, declared inference inputs and output structure. Model execution is documented in the execution receipts.'}
    finally:
        if archive is not None:
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).parent)
    print(json.dumps(verify(parser.parse_args().package), indent=2))
