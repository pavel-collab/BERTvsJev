"""Оценить сохранённый BERT на validation или test."""
import argparse
import json
import importlib.metadata
import time

import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import LABELS, ROOT, device_for, metrics, read_rows, settings
from telemetry import Telemetry, latency_summary

WARMUP_BATCHES = 5


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def save_confusion_matrix(matrix, task, labels, split, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {
        "urgent": ["Not urgent", "Urgent"],
        "sentiment": ["Negative", "Neutral", "Positive"],
    }.get(task, [str(label) for label in labels])
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay(matrix, display_labels=names).plot(
        ax=axis, cmap="Blues", values_format="d", colorbar=False,
    )
    axis.set_title(f"{split}: {task} (counts)")
    figure.savefig(output / f"{split}_{task}_confusion_matrix.png", dpi=160)
    plt.close(figure)


def print_metrics(report, split, output):
    print(f"\nВыборка: {split}; устройство: {report['device']}")
    print(f"{'Задача':<12} {'N':>5} {'Accuracy':>10} {'Macro-F1':>10} {'Precision*':>12} {'Recall*':>10}")
    for task, result in report["tasks"].items():
        print(f"{task:<12} {result['count']:>5} {result['accuracy']:>10.4f} "
              f"{result['macro_f1']:>10.4f} {result['macro_precision']:>12.4f} "
              f"{result['macro_recall']:>10.4f}")
    print("* Precision и recall усреднены по классам (macro).")
    print(f"Отчёты, предсказания и матрицы ошибок: {output}")


def evaluate(split, batch_size=None, warmup=WARMUP_BATCHES):
    config = settings()
    if batch_size is not None:
        config["batch_size"] = batch_size
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
    output = config["results_dir"]
    output.mkdir(parents=True, exist_ok=True)

    telemetry = Telemetry("bert", dataset, output, {
        "split": split, "device": str(device), "batch_size": config["batch_size"],
        "warmup_batches_per_task": warmup, "max_length": config["max_length"],
        "expected_examples": len(rows), "torch_threads": torch.get_num_threads(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "training_metadata": metadata,
        "latency_definition": "Batch: tokenization, device transfers, inference and probabilities on CPU; synchronized; excludes loading, warmup and metric computation",
        "cost": "Local infrastructure cost unknown",
    })
    completed = False
    try:
        for task, labels in LABELS.items():
            folder = config["model_dir"] / task
            load_started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True).to(device).eval()
            synchronize(device)
            telemetry.metadata.setdefault("model_loading_seconds", {})[task] = time.perf_counter() - load_started
            truth, predicted = [], []
            with torch.inference_mode():
                subsets = [rows[start:start + config["batch_size"]] for start in range(0, len(rows), config["batch_size"])]
                schedule = [("warmup", i, subsets[i % len(subsets)]) for i in range(warmup)] if subsets else []
                schedule += [("measured", i, subset) for i, subset in enumerate(subsets)]
                for phase, batch_index, subset in schedule:
                    synchronize(device)
                    started = time.perf_counter()
                    try:
                        inputs = tokenizer([row["text"] for row in subset], padding=True, truncation=True,
                                           max_length=config["max_length"], return_tensors="pt")
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        probabilities = model(**inputs).logits.softmax(-1).cpu().tolist()
                        synchronize(device)
                    except Exception as exc:
                        telemetry.record(task=task, phase=phase, batch_index=batch_index, status="error",
                                         ids=[r["id"] for r in subset], examples=len(subset),
                                         latency_seconds=time.perf_counter() - started,
                                         error_type=type(exc).__name__, error=str(exc))
                        raise
                    elapsed = time.perf_counter() - started
                    lengths = [len(tokenizer(row["text"], truncation=False)["input_ids"]) for row in subset]
                    telemetry.record(task=task, phase=phase, batch_index=batch_index, status="ok",
                                     ids=[r["id"] for r in subset], examples=len(subset), latency_seconds=elapsed,
                                     input_characters=[len(r["text"]) for r in subset], untruncated_tokens=lengths,
                                     truncated=[n > config["max_length"] for n in lengths],
                                     padded_sequence_length=inputs["input_ids"].shape[1])
                    if phase == "warmup":
                        continue
                    for row, probs in zip(subset, probabilities):
                        prediction = max(range(len(probs)), key=probs.__getitem__)
                        predicted.append(prediction)
                        truth.append(labels.index(row["labels"][task]))
                        records[row["id"]]["predictions"][task] = labels[prediction]
                        records[row["id"]]["probabilities"][task] = {str(label): p for label, p in zip(labels, probs)}
            report["tasks"][task] = metrics(truth, predicted, labels)
            report["tasks"][task]["inference_seconds"] = sum(r["latency_seconds"] for r in telemetry.records if r["task"] == task and r["phase"] == "measured")
            matrix = confusion_matrix(truth, predicted, labels=list(range(len(labels))))
            report["tasks"][task]["confusion_matrix"] = matrix.tolist()
            report["tasks"][task]["class_order"] = labels
            save_confusion_matrix(matrix, task, labels, split, output)
            del model
        (output / f"{split}_predictions.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records.values()))
        (output / f"{split}_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print_metrics(report, split, output)
        completed = True
    finally:
        task_reports = {}
        combined = {}
        for task in LABELS:
            measured = [r for r in telemetry.records if r["task"] == task and r["phase"] == "measured" and r["status"] == "ok"]
            task_reports[task] = latency_summary([r["latency_seconds"] for r in measured], sum(r["examples"] for r in measured))
            for record in measured:
                batch = combined.setdefault(record["batch_index"], {"tasks": set(), "seconds": 0, "examples": record["examples"]})
                batch["tasks"].add(task)
                batch["seconds"] += record["latency_seconds"]
        full = [b for b in combined.values() if b["tasks"] == set(LABELS)]
        telemetry.finish(completed, {"tasks": task_reports,
            "three_tasks_batch_latency": latency_summary([b["seconds"] for b in full], sum(b["examples"] for b in full)),
            "three_tasks_definition": "Sum of separately measured task batches with matching IDs; not a simultaneous end-to-end service call",
            "estimated_total_cost_usd": None})



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("validation", "test"), nargs="?", default="test",
                        help="Выборка из настроек train.py (по умолчанию test)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP_BATCHES)
    args = parser.parse_args()
    if args.warmup < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("warmup >= 0, batch-size >= 1")
    evaluate(args.split, args.batch_size, args.warmup)


if __name__ == "__main__":
    main()
