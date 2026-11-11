from collections import defaultdict
def fuse_occurrences(by_modality):
    locations = defaultdict(list)
    for modality, rows in by_modality.items():
        for row in rows:
            locations[row['large_unit'], int(row['offset'])].append({**row, 'modality': modality})
    fused = []
    conflicts = []
    for (unit, offset), rows in sorted(locations.items()):
        keys = {r['small_key'] for r in rows}
        if len(keys) != 1:
            conflicts.append(dict(large_unit=unit, offset=offset, small_keys=sorted(keys)))
            continue
        modalities = sorted({r['modality'] for r in rows})
        fused.append(dict(large_unit=unit, offset=offset, small_key=next(iter(keys)), modalities=modalities, cross_modal_confirmed=len(modalities) > 1, source_scores={r['modality']: r['score'] for r in rows}))
    return (fused, conflicts)
