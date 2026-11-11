"""Action and sequence routing from matched clips, using the recovered rules.

Inputs are questions, clip detections and recording identities. No historical
answer vector or test-ID override table is read by this module.
"""
from collections import defaultdict
from itertools import product
import re
LETTERS = 'ABCD'

def atoms(value):
    return tuple((' '.join(part.casefold().split()) for part in str(value).split(',') if part.strip()))

def action_answer(row, evidence, parent, minimum_actions=4, allow_partial=True):
    options = {ch: atoms(row[ch]) for ch in LETTERS}
    category = row['category']
    singleton = [ch for ch, values in options.items() if len(values) == 1 and values[0] in evidence]
    if category == 'single':
        return (singleton[0] if len(singleton) == 1 else parent, dict(rule='unique_single', supported=singleton))
    if category == 'multi':
        full = ''.join(singleton)
        replace = len(evidence) >= minimum_actions and bool(full)
        answer = full if replace else ''.join((ch for ch in LETTERS if ch in parent or ch in full))
        return (answer, dict(rule='full_multi' if replace else 'add_multi', supported=singleton, minimum_actions=minimum_actions))
    if category != 'combination':
        return (parent, dict(rule='unchanged_category'))
    strict = [ch for ch, values in options.items() if values and set(values) <= evidence]
    if len(strict) == 1:
        return (strict[0], dict(rule='complete_combination', supported=strict))
    if not allow_partial:
        return (parent, dict(rule='combination_abstain'))
    counts = {ch: len(set(values) & evidence) for ch, values in options.items()}
    fractions = {ch: counts[ch] / len(set(values)) if values else 0.0 for ch, values in options.items()}
    best, runner = sorted(LETTERS, key=lambda ch: (-counts[ch], -fractions[ch], ch))[:2]
    passed = counts[best] >= 4 and fractions[best] >= 0.5 and (counts[best] - counts[runner] >= 2)
    return (best if passed else parent, dict(rule='partial_combination' if passed else 'combination_abstain', counts=counts, fractions=fractions))

def eligible_clips(timelines):
    result = {}
    for unit, clips in timelines.items():
        selected = []
        for clip in clips:
            modalities = {str(value).casefold() for value in clip['modalities']}
            scores = clip.get('source_scores')
            if scores is None:
                raise ValueError('Routing requires separate raw matching scores for each modality')
            if 'depth_color' not in modalities and float(scores.get('ir', 0.0)) < 1.5:
                continue
            selected.append(dict(clip, quality=max(scores.values()), cross_modal_confirmed=bool(clip.get('cross_modal_confirmed', len(modalities) > 1))))
        result[unit] = selected
    return result
