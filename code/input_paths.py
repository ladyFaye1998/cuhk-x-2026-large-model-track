"""Locate the four official input sources beneath an organizer data directory."""
from pathlib import Path

def resolve_data(root):
    root=Path(root).resolve(strict=True)
    if not root.is_dir():raise ValueError('Data root must be a directory')
    def unique(patterns, label):
        matches=sorted({p.resolve() for pattern in patterns for p in root.rglob(pattern) if p.is_file()})
        if len(matches)!=1:raise ValueError(f'Expected one {label}; found {len(matches)}')
        return matches[0]
    return {'questions':unique(['test_qa.csv'],'question table'),
            'videos':unique(['large_model_track_test.zip','large_model_track_test.zip.001'],'Large video archive'),
            'sensors':unique(['large_sensors.zip','large_sensors.zip.001'],'Large sensor archive'),
            'small_archive':unique(['small_model_track_test.zip','small_model_track_test.zip.001'],'atomic clip archive')}


def prepare_sensors(source, output):
    """Expose ZIP members beneath the official Testing directory convention."""
    import shutil,zipfile
    source=Path(source).resolve(strict=True)
    if source.is_dir():return source
    output=Path(output).resolve()/'Testing'
    output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir():continue
            destination=(output/member.filename).resolve()
            if not destination.is_relative_to(output):raise ValueError('Invalid raw archive path')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(member) as raw,destination.open('wb') as target:shutil.copyfileobj(raw,target)
    return output
