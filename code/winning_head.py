"""Apply the recovered question-aware classifier to a newly generated raw-model run."""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re

import joblib


LETTERS = set("ABCD")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_text(row: dict[str, str], raw_prediction: str) -> str:
    fields = [
        row["source"], row["category"], row["question"],
        "option_a", row["A"], "option_b", row["B"],
        "option_c", row["C"], "option_d", row["D"],
        "raw_model", raw_prediction,
    ]
    return re.sub(r"\s+", " ", " ".join(fields).casefold()).strip()


def predict(questions: list[dict[str, str]], raw_rows: list[dict[str, str]], model_path: Path) -> list[str]:
    ids = [row["qa_id"] for row in questions]
    if ids != [row["qa_id"] for row in raw_rows]:
        raise ValueError("Raw-model output does not match the question order")
    if len(ids) != len(set(ids)):
        raise ValueError("Question identifiers must be unique")
    if any(row.get(name, "").strip() for row in questions for name in ("answer", "target", "label", "prediction")):
        raise ValueError("Question input contains an answer-bearing field")
    package = joblib.load(model_path)
    if package.get("schema") != "cuhkx-large-question-aware-head/v1":
        raise ValueError("Unsupported question-aware model")
    contract = package["feature_contract"]
    if contract.get("excludes") != ["qa_id", "path"]:
        raise ValueError("Unexpected model feature contract")
    documents = [normalized_text(row, raw["prediction"]) for row, raw in zip(questions, raw_rows, strict=True)]
    features = package["vectorizer"].transform(documents)
    predictions = package["classifier"].predict(features).tolist()
    for row, answer in zip(questions, predictions, strict=True):
        if not answer or not set(answer) <= LETTERS or len(answer) != len(set(answer)):
            raise ValueError("Classifier produced an invalid answer")
        if row["category"] == "sequence" and set(answer) != LETTERS:
            raise ValueError("Classifier produced an incomplete sequence")
        if row["category"] not in ("sequence", "multi") and len(answer) != 1:
            raise ValueError("Classifier produced a non-singular answer")
    return predictions


def write(path: Path, questions: list[dict[str, str]], predictions: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(["qa_id", "prediction"])
        writer.writerows((row["qa_id"], answer) for row, answer in zip(questions, predictions, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = read(args.questions)
    raw_rows = read(args.raw_predictions)
    predictions = predict(questions, raw_rows, args.model)
    write(args.output, questions, predictions)
    report = {
        "technical_verification": "PASS",
        "rows": len(predictions),
        "raw_model_rows": len(raw_rows),
        "prediction_csv_inputs": [],
        "identifier_features": [],
        "path_features": [],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
