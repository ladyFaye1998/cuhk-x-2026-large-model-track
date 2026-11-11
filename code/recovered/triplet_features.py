"""Original triplet option features, extracted without experiment I/O.

Source: explore_triplet_joint_text_decoder.py
SHA256: 3d083130852f00920cf66b61464e2f9db4aac55cb3608bc84c64d92f3bfd5582
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
LETTERS = "ABCD"
ACTION_CATEGORIES = {"single", "multi", "combination", "sequence"}

def norm(text: str) -> str:
    return " ".join(str(text).casefold().strip().split())


def atoms(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(norm(x) for x in str(text).split(",") if norm(x)))


def build_context(rows: list[dict[str, str]]) -> dict[str, object]:
    occurrences: list[tuple[str, str, int, str, int]] = []
    # atom, category, pace, qa_id, option index
    for row in rows:
        if row["category"] not in ACTION_CATEGORIES:
            continue
        pace = int(row["_pace"])
        for option_index, letter in enumerate(LETTERS):
            for atom in atoms(row[letter]):
                occurrences.append((atom, row["category"], pace, row["qa_id"], option_index))

    total = Counter(a for a, *_ in occurrences)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_pace: dict[int, Counter[str]] = defaultdict(Counter)
    category_support: dict[str, set[str]] = defaultdict(set)
    pace_support: dict[str, set[int]] = defaultdict(set)
    row_support: dict[str, set[str]] = defaultdict(set)
    option_support: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for atom, category, pace, qa_id, option_index in occurrences:
        by_category[category][atom] += 1
        by_pace[pace][atom] += 1
        category_support[atom].add(category)
        pace_support[atom].add(pace)
        row_support[atom].add(qa_id)
        option_support[atom].add((qa_id, option_index))
    return {
        "total": total,
        "by_category": by_category,
        "by_pace": by_pace,
        "category_support": category_support,
        "pace_support": pace_support,
        "row_support": row_support,
        "option_support": option_support,
    }


def stats(values: list[float], prefix: str, out: dict[str, float]) -> None:
    if not values:
        for name in ("sum", "mean", "min", "max", "std"):
            out[f"{prefix}_{name}"] = 0.0
        return
    arr = np.asarray(values, dtype=np.float64)
    out[f"{prefix}_sum"] = float(arr.sum())
    out[f"{prefix}_mean"] = float(arr.mean())
    out[f"{prefix}_min"] = float(arr.min())
    out[f"{prefix}_max"] = float(arr.max())
    out[f"{prefix}_std"] = float(arr.std())


def option_features(row: dict[str, str], letter: str, context: dict[str, object]) -> dict[str, float]:
    category = row["category"]
    pace = int(row["_pace"])
    aa = atoms(row[letter])
    total: Counter[str] = context["total"]  # type: ignore[assignment]
    by_category: dict[str, Counter[str]] = context["by_category"]  # type: ignore[assignment]
    by_pace: dict[int, Counter[str]] = context["by_pace"]  # type: ignore[assignment]
    category_support: dict[str, set[str]] = context["category_support"]  # type: ignore[assignment]
    pace_support: dict[str, set[int]] = context["pace_support"]  # type: ignore[assignment]
    row_support: dict[str, set[str]] = context["row_support"]  # type: ignore[assignment]
    option_support: dict[str, set[tuple[str, int]]] = context["option_support"]  # type: ignore[assignment]

    out: dict[str, float] = {
        "atom_count": float(len(aa)),
        "pace": float(pace),
        "letter_index": float(LETTERS.index(letter)),
        "row_option_atom_count_mean": float(np.mean([len(atoms(row[x])) for x in LETTERS])),
        "row_option_atom_count_max": float(max(len(atoms(row[x])) for x in LETTERS)),
    }
    stats([total[a] for a in aa], "total", out)
    stats([total[a] - sum(a in atoms(row[x]) for x in LETTERS) for a in aa], "other_rows", out)
    stats([by_category[category][a] for a in aa], "same_category", out)
    stats([by_pace[pace][a] for a in aa], "same_pace", out)
    stats([len(category_support[a]) for a in aa], "category_support", out)
    stats([len(pace_support[a]) for a in aa], "pace_support", out)
    stats([len(row_support[a]) for a in aa], "row_support", out)
    stats([len(option_support[a]) for a in aa], "option_support", out)

    for cat in sorted(ACTION_CATEGORIES):
        stats([by_category[cat][a] for a in aa], f"cat_{cat}", out)
    for p in (1, 2, 3):
        stats([by_pace[p][a] for a in aa], f"pace_{p}", out)

    option_sets = {x: set(atoms(row[x])) for x in LETTERS}
    mine = option_sets[letter]
    overlaps = [len(mine & option_sets[x]) for x in LETTERS if x != letter]
    out["within_row_overlap_sum"] = float(sum(overlaps))
    out["within_row_overlap_max"] = float(max(overlaps) if overlaps else 0)
    out["within_row_unique_atoms"] = float(sum(all(a not in option_sets[x] for x in LETTERS if x != letter) for a in mine))
    return out


def vectorize(feature_rows: list[dict[str, float]], names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if names is None:
        names = sorted({key for row in feature_rows for key in row})
    return np.asarray([[row.get(name, 0.0) for name in names] for row in feature_rows], dtype=np.float32), names
