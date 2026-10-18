"""Чтение данных, проверки выборок и метрики классификации GRU-CNN."""
import hashlib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/authored-v2-en"
LABELS = {"topic": ["billing", "technical", "sales", "other"],
          "urgent": [0, 1], "sentiment": [0, 1, 2]}


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Пустая выборка: {path}")
    ids = set()
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Каждая запись должна содержать непустой text")
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Каждая запись должна иметь уникальный строковый id")
        ids.add(row["id"])
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError("Отсутствует group_id")
        for task, labels in LABELS.items():
            if row.get("labels", {}).get(task) not in labels:
                raise ValueError(f"Неизвестная метка {task}: {row['id']}")
    return rows


def check_disjoint(left, right):
    for field in ("id", "group_id"):
        if {row[field] for row in left} & {row[field] for row in right}:
            raise ValueError(f"Пересечение выборок по {field}")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def metrics(truth, predicted, labels):
    names = [str(label) for label in labels]
    report = classification_report(truth, predicted, labels=labels,
                                   target_names=names, output_dict=True, zero_division=0)
    return {
        "count": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "per_class": {name: {"precision": report[name]["precision"],
                             "recall": report[name]["recall"],
                             "f1": report[name]["f1-score"],
                             "support": int(report[name]["support"])} for name in names},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "class_order": labels,
    }

