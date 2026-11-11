"""Original reciprocal-neighbor infrared grouping; no prediction parent I/O."""
from __future__ import annotations
import hashlib,re
from dataclasses import dataclass
import numpy as np

def embedding_key(vector: np.ndarray) -> str:
    raw = np.ascontiguousarray(vector.astype('<f4', copy=False))
    return hashlib.sha256(raw.tobytes()).hexdigest()

@dataclass(frozen=True)
class Triple:
    members: tuple[str, str, str]
    member_keys: tuple[str, str, str]
    minimum_cosine: float
    reciprocal_margin: float
    correct_group_diagnostic_only: bool | None

def recover_triplets(unit_names: list[str], embeddings: np.ndarray, pool_indices: list[int], training_diagnostic: bool) -> list[Triple]:
    indices = list(pool_indices)
    if len(indices) < 4:
        return []
    keys = [embedding_key(embeddings[index]) for index in indices]
    if len(keys) != len(set(keys)):
        raise RuntimeError('embedding-key collision')
    order = sorted(range(len(indices)), key=lambda offset: keys[offset])
    indices = [indices[offset] for offset in order]
    keys = [keys[offset] for offset in order]
    matrix = np.asarray(embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError('invalid embedding norm')
    matrix /= norms
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    top_two = []
    for left in range(len(indices)):
        ranked = sorted((right for right in range(len(indices)) if right != left), key=lambda right: (-float(similarity[left, right]), keys[right]))
        top_two.append((ranked[0], ranked[1]))
    output = []
    seen = set()
    for left in range(len(indices)):
        candidate = tuple(sorted((left, *top_two[left])))
        if len(set(candidate)) != 3 or candidate in seen:
            continue
        if not all((set(top_two[item]) == set(candidate) - {item} for item in candidate)):
            continue
        seen.add(candidate)
        pair_scores = [float(similarity[candidate[i], candidate[j]]) for i in range(3) for j in range(i + 1, 3)]
        margins = []
        for item in candidate:
            weak_internal = min((float(similarity[item, other]) for other in candidate if other != item))
            best_external = max((float(similarity[item, other]) for other in range(len(indices)) if other not in candidate))
            margins.append(weak_internal - best_external)
        members_with_keys = sorted(((keys[item], unit_names[indices[item]]) for item in candidate), key=lambda value: value[0])
        members = tuple((str(value[1]) for value in members_with_keys))
        diagnostic = None
        if training_diagnostic:
            diagnostic = len({re.sub('-[123]$', '', member) for member in members}) == 1
        output.append(Triple(members=members, member_keys=tuple((str(value[0]) for value in members_with_keys)), minimum_cosine=min(pair_scores), reciprocal_margin=min(margins), correct_group_diagnostic_only=diagnostic))
    return sorted(output, key=lambda triple: triple.member_keys)
