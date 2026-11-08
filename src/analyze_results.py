"""Score complete paired runs and calculate the pre-specified statistics."""

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from a3_metrics import mcnemar_exact, normalize, score, wilson
from protocol import CONDITIONS, RETRIEVAL, ROOT, data_directory, json_data, load_data, sha256, write_json


def scored_rows(run, metadata, questions, answers):
    if metadata["status"] != "completed":
        raise ValueError("The run must be complete before it can be scored.")
    prediction_bytes = (run / "predictions.jsonl").read_bytes()
    if hashlib.sha256(prediction_bytes).hexdigest() != metadata["predictions_sha256"]:
        raise ValueError("The predictions do not match the completed run metadata.")
    rows = []
    seen = set()
    for line in prediction_bytes.splitlines():
        prediction = json.loads(line)
        question_id, condition = prediction["id"], prediction["condition"]
        key = (question_id, condition)
        if question_id not in questions or condition not in CONDITIONS or key in seen:
            raise ValueError(f"Invalid or duplicate prediction {key}.")
        question = questions[question_id]
        if any(prediction[field] != question[field] for field in ("question", "repository", "split")):
            raise ValueError(f"Prediction does not match the dataset question {question_id}.")
        seen.add(key)
        row = score(prediction, answers[question_id])
        if condition not in RETRIEVAL:
            if prediction["retrieved"]:
                raise ValueError(f"Closed-book prediction {question_id} contains retrieved documents.")
            for metric in ("supported_by_target_record", "unsupported_tag", "gold_in_retrieved_context", "target_record_retrieved"):
                row[metric] = None
            if row["error_category"] == "unsupported_tag":
                row["error_category"] = "incorrect_tag"
        rows.append(row)
    expected = {(question_id, condition) for question_id in questions for condition in CONDITIONS}
    if seen != expected or metadata["n_predictions"] != len(rows):
        raise ValueError("Expected one prediction for every question and condition.")
    return rows


def condition_summary(rows, uses_retrieval):
    correct = sum(row["correct"] for row in rows)
    times = [row["elapsed_seconds"] for row in rows]
    q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive") if len(times) > 1 else [times[0]] * 3
    lower, upper = wilson(correct, len(rows))
    result = {
        "n": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "wilson_95_ci": [max(0.0, lower), min(1.0, upper)],
        "latency_seconds": {"median": statistics.median(times), "q1": q1, "q3": q3, "iqr": q3 - q1},
        "error_categories": dict(Counter(row["error_category"] for row in rows if not row["correct"])),
        "retrieval_metrics": None,
    }
    if uses_retrieval:
        result["retrieval_metrics"] = {
            "target_record_recall_at_3": sum(row["target_record_retrieved"] for row in rows) / len(rows),
            "gold_evidence_coverage_at_3": sum(row["gold_in_retrieved_context"] for row in rows) / len(rows),
            "unsupported_tag_count": sum(row["unsupported_tag"] for row in rows),
        }
    return result


def paired_comparison(updated, comparator, question_ids, bootstrap=False):
    pairs = [(updated[q]["correct"], comparator[q]["correct"]) for q in question_ids]
    diffs = [int(a) - int(b) for a, b in pairs]
    favorable, unfavorable = diffs.count(1), diffs.count(-1)
    result = {
        "updated_only_correct": favorable, "comparator_only_correct": unfavorable,
        "both_correct": sum(a and b for a, b in pairs),
        "both_incorrect": sum(not a and not b for a, b in pairs),
        "accuracy_difference_percentage_points": 100 * sum(diffs) / len(diffs),
        "mcnemar_exact_two_sided_p": mcnemar_exact(favorable, unfavorable),
    }
    if bootstrap:
        rng = random.Random(42)
        samples = sorted(100 * sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
        result["paired_bootstrap_95_ci_percentage_points"] = [samples[249], samples[9749]]
    return result


def summarize(rows, answers, dataset):
    by = {condition: {} for condition in CONDITIONS}
    for row in rows:
        by[row["condition"]][row["id"]] = row
    question_ids = sorted(answers)
    updated = by["tfidf_updated"]
    primary = paired_comparison(updated, by["tfidf_frozen"], question_ids, bootstrap=True)
    primary["supports_h1"] = (
        primary["accuracy_difference_percentage_points"] > 0 and primary["mcnemar_exact_two_sided_p"] < 0.05
    )
    secondary = {
        condition: paired_comparison(updated, by[condition], question_ids)
        for condition in ("closed_book", "bm25_updated")
    }
    ordered = sorted(secondary.values(), key=lambda comparison: comparison["mcnemar_exact_two_sided_p"])
    previous = 0.0
    for index, comparison in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * comparison["mcnemar_exact_two_sided_p"]))
        comparison["holm_adjusted_p"] = previous
    strata = {}
    for name, changed in (("changed", True), ("unchanged", False)):
        ids = [q for q in question_ids if answers[q]["changed"] == changed]
        strata[name] = {"n": len(ids), "correct": {
            condition: sum(group[q]["correct"] for q in ids) for condition, group in by.items()
        }}
    return {
        "dataset": dataset, "n_questions": len(question_ids), "n_predictions": len(rows),
        "conditions": {
            condition: condition_summary(list(group.values()), condition in RETRIEVAL)
            for condition, group in by.items()
        },
        "primary_tfidf_updated_vs_frozen": primary,
        "secondary_tfidf_updated_vs": secondary,
        "strata": strata,
    }


def analyze(args, directory):
    metadata = json_data(args.run / "run_metadata.json")
    if metadata["dataset"] != args.dataset:
        raise ValueError("The requested dataset does not match the run.")
    data, hashes = load_data(directory, args.dataset, ("questions.json", "answer_key.json"))
    for name in hashes.keys() - {"answer_key.json"}:
        if hashes[name] != metadata["inputs_sha256"][name]:
            raise ValueError(f"{name} does not match the dataset used for the run.")
    questions = {q["id"]: q for q in data["questions.json"]}
    answers = {a["id"]: a for a in data["answer_key.json"]}
    if (
        not questions or questions.keys() != answers.keys()
        or len(questions) != len(data["questions.json"])
        or len(answers) != len(data["answer_key.json"])
    ):
        raise ValueError("Expected one answer key entry per question.")
    if any(answer["changed"] != (normalize(answer["old_tag"]) != normalize(answer["answer_tag"])) for answer in answers.values()):
        raise ValueError("The answer key has an inconsistent changed-tag flag.")
    rows = scored_rows(args.run, metadata, questions, answers)
    summary = summarize(rows, answers, args.dataset)
    summary["analysis_inputs_sha256"] = {**hashes, "predictions.jsonl": metadata["predictions_sha256"]}
    summary["analysis_source_sha256"] = {
        name: sha256(ROOT / "src" / name) for name in ("analyze_results.py", "a3_metrics.py", "protocol.py")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "scored_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "summary.json", summary)
    for condition, result in summary["conditions"].items():
        print(f"{condition}: {result['correct']}/{result['n']} correct ({result['accuracy']:.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "holdout"), default="pilot")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args, data_directory(parser, args))


if __name__ == "__main__":
    main()
