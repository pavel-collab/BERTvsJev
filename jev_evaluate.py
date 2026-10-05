"""Преобразовать сохранённые ответы Jev в классификацию и оценить без API."""
import argparse
import json
import math
from pathlib import Path
import sys

from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "bert"))
from train import LABELS, TEST_FILE, VALIDATION_FILE, metrics
from test import save_confusion_matrix
from jev import ROOT, read_rows


def read_results(path):
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError("Ожидается непустой JSON-массив или JSONL с ответами Jev")
    indexed = {}
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or identifier in indexed:
            raise ValueError(f"Отсутствующий или повторяющийся id: {identifier!r}")
        indexed[identifier] = record
    return indexed


def probability(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"Некорректная вероятность: {value!r}")
    return float(value)


def convert(record, threshold):
    answers = record["response"]["answers"]
    predictions, probabilities = {}, {}
    for task, labels in LABELS.items():
        answer = answers[task]
        if task == "urgent":
            score = probability(answer["noul"])
            probs = {"0": 1 - score, "1": score}
            prediction = int(score >= threshold)
        else:
            probs = {str(label): probability(answer["probabilities"][str(label)]) for label in labels}
            if not math.isclose(sum(probs.values()), 1, abs_tol=0.025):
                raise ValueError(f"{task}: сумма вероятностей отличается от 1")
            # API округляет вероятности; нормализация нужна для multiclass ROC-AUC.
            total = sum(probs.values())
            probs = {key: value / total for key, value in probs.items()}
            prediction = (answer["choice"] if task == "topic"
                          else max(labels, key=lambda label: probs[str(label)]))
            if prediction not in labels:
                raise ValueError(f"{task}: неизвестный класс {prediction!r}")
        predictions[task], probabilities[task] = prediction, probs
    return predictions, probabilities


def auc_metrics(truth, probabilities, labels):
    per_class = {}
    for index, label in enumerate(labels):
        binary = [int(value == index) for value in truth]
        per_class[str(label)] = (float(roc_auc_score(binary, [row[index] for row in probabilities]))
                                 if len(set(binary)) == 2 else None)
    if len(labels) == 2:
        auc = per_class[str(labels[1])]
    else:
        auc = (sum(per_class.values()) / len(labels)
               if all(value is not None for value in per_class.values()) else None)
    return {"roc_auc": auc, "roc_auc_per_class": per_class,
            "roc_auc_method": "binary_positive_class" if len(labels) == 2 else "ovr_macro",
            "roc_auc_note": None if auc is not None else "Для ROC-AUC нужны оба исхода каждого класса"}


def evaluate(input_path, dataset, output, split="test", threshold=0.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Порог срочности должен быть в [0, 1]")
    rows = read_rows(dataset)
    raw = read_results(input_path)
    ids = {row["id"] for row in rows}
    if ids != set(raw):
        raise ValueError(f"ID ответов и датасета не совпадают: отсутствуют {sorted(ids - set(raw))}; "
                         f"лишние {sorted(set(raw) - ids)}")
    records = []
    for row in rows:
        try:
            predictions, probabilities = convert(raw[row["id"]], threshold)
            for task, labels in LABELS.items():
                if row["labels"][task] not in labels:
                    raise ValueError(f"{task}: неизвестная истинная метка")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Запись {row['id']}: {exc}") from exc
        records.append({"id": row["id"], "labels": row["labels"],
                        "predictions": predictions, "probabilities": probabilities})
    report = {"dataset": str(dataset), "device": "remote", "source": str(input_path),
              "urgent_threshold": threshold, "tasks": {}}
    matrices = {}
    for task, labels in LABELS.items():
        truth = [labels.index(row["labels"][task]) for row in records]
        predicted = [labels.index(row["predictions"][task]) for row in records]
        probs = [[row["probabilities"][task][str(label)] for label in labels] for row in records]
        result = metrics(truth, predicted, labels)
        result.update(auc_metrics(truth, probs, labels))
        matrices[task] = confusion_matrix(truth, predicted, labels=list(range(len(labels))))
        result.update(confusion_matrix=matrices[task].tolist(), class_order=labels)
        report["tasks"][task] = result
    output.mkdir(parents=True, exist_ok=True)
    for task, matrix in matrices.items():
        save_confusion_matrix(matrix, task, LABELS[task], split, output)
    (output / f"{split}_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    (output / f"{split}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{'Задача':<12} {'N':>5} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}")
    for task, result in report["tasks"].items():
        auc = f"{result['roc_auc']:.4f}" if result["roc_auc"] is not None else "n/a"
        print(f"{task:<12} {result['count']:>5} {result['accuracy']:>10.4f} {result['macro_f1']:>10.4f} {auc:>10}")
    print(f"Результаты: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=ROOT / "results/jev_test_run02.json")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--dataset", type=Path, help="JSONL с истинными метками")
    parser.add_argument("--output", type=Path, default=ROOT / "results/jev-en")
    parser.add_argument("--urgent-threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        evaluate(args.input, args.dataset or (TEST_FILE if args.split == "test" else VALIDATION_FILE),
                 args.output, args.split, args.urgent_threshold)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
