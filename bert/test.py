"""Оценить сохранённый BERT на validation или test."""
import argparse
import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import LABELS, ROOT, BatchGenerator, device_for, metrics, read_rows, settings


def evaluate(split):
    config = settings()
    dataset = config[f"{split}_file"]
    rows = read_rows(dataset)
    metadata = json.loads((config["model_dir"] / "training.json").read_text())

    # Настройки токенизации должны совпадать с обучением.
    config["max_length"] = metadata["max_length"]
    if metadata["labels"] != LABELS:
        raise ValueError("Метки сохранённой модели отличаются от текущей схемы")

    if split == "test":
        for name in ("train", "validation"):
            reference = ROOT / metadata[f"{name}_file"]
            original = read_rows(reference)
            for field in ("id", "group_id"):
                if {r[field] for r in original} & {r[field] for r in rows}:
                    raise ValueError(f"Пересечение {name}/test по {field}")
            
    device = device_for(config)
    records = {r["id"]: {"id": r["id"], "labels": r["labels"], "predictions": {}, "probabilities": {}} for r in rows}
    report = {"dataset": str(dataset), "device": str(device), "tasks": {}}

    for task, labels in LABELS.items():
        folder = config["model_dir"] / task
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True).to(device).eval()
        truth, predicted = [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for subset, inputs in BatchGenerator(rows, tokenizer, config, device):
                probabilities = model(**inputs).logits.softmax(-1).cpu().tolist()
                for row, probs in zip(subset, probabilities):
                    prediction = max(range(len(probs)), key=probs.__getitem__)
                    predicted.append(prediction)
                    truth.append(labels.index(row["labels"][task]))
                    records[row["id"]]["predictions"][task] = labels[prediction]
                    records[row["id"]]["probabilities"][task] = {str(label): p for label, p in zip(labels, probs)}
        report["tasks"][task] = metrics(truth, predicted, labels)
        report["tasks"][task]["inference_seconds"] = time.perf_counter() - started
        del model
    output = config["results_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_predictions.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records.values()))
    (output / f"{split}_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("validation", "test"), nargs="?", default="test",
                        help="Выборка из настроек train.py (по умолчанию test)")
    args = parser.parse_args()
    evaluate(args.split)


if __name__ == "__main__":
    main()
