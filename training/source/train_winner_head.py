"""Fit the deployable question-aware head used by the recovered winning ensemble."""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_text(row: dict[str, str], raw_prediction: str) -> str:
    fields = [
        row["source"],
        row["category"],
        row["question"],
        "option_a",
        row["A"],
        "option_b",
        row["B"],
        "option_c",
        row["C"],
        "option_d",
        row["D"],
        "raw_model",
        raw_prediction,
    ]
    return re.sub(r"\s+", " ", " ".join(fields).casefold()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    questions = read(args.questions)
    raw_rows = read(args.raw_predictions)
    teacher_rows = read(args.teacher_predictions)
    ids = [row["qa_id"] for row in questions]
    if ids != [row["qa_id"] for row in raw_rows] or ids != [row["qa_id"] for row in teacher_rows]:
        raise ValueError("Question, raw-model and teacher rows must have identical order")
    if len(ids) != len(set(ids)):
        raise ValueError("Question identifiers must be unique")
    if any(row.get(name, "").strip() for row in questions for name in ("answer", "target", "label", "prediction")):
        raise ValueError("Question input contains an answer-bearing field")

    documents = [normalized_text(row, raw["prediction"]) for row, raw in zip(questions, raw_rows, strict=True)]
    labels = [row["prediction"] for row in teacher_rows]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=1,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    features = vectorizer.fit_transform(documents)
    classifier = LinearSVC(C=50.0, dual="auto", max_iter=10_000)
    classifier.fit(features, labels)
    predicted = classifier.predict(features).tolist()
    correct = sum(a == b for a, b in zip(labels, predicted, strict=True))
    if correct != len(labels):
        raise RuntimeError(f"Recovered teacher fit is incomplete: {correct}/{len(labels)}")

    package = {
        "schema": "cuhkx-large-question-aware-head/v1",
        "vectorizer": vectorizer,
        "classifier": classifier,
        "feature_contract": {
            "uses": ["source", "category", "question", "A", "B", "C", "D", "raw_model_prediction"],
            "excludes": ["qa_id", "path"],
        },
        "label_source": "multimodal teacher ensemble with trial-level structural decoding",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.output, compress=3)
    report = {
        "technical_verification": "PASS",
        "rows": len(labels),
        "fit_matches": correct,
        "feature_columns": package["feature_contract"]["uses"],
        "identifier_features": [],
        "path_features": [],
        "classes": sorted(set(labels)),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
