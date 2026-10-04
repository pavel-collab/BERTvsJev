"""Fine-tune трёх классификаторов Transformers: topic, urgent, sentiment."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev import ROOT, read_rows

import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

BERT_MODEL = 'DeepPavlov/rubert-base-cased'
BERT_REVISION = 'main'
BERT_OUTPUT_DIR = ROOT / 'bert/weights'
TRAIN_FILE = ROOT / 'data/authored-v2/train.jsonl'
VALIDATION_FILE = ROOT / 'data/authored-v2/validation.jsonl'
TEST_FILE = ROOT / 'data/authored-v2/test.jsonl'
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 256
BERT_LEARNING_RATE = 2e-05
BERT_SEED = 42
BERT_DEVICE = 'auto'
BERT_RESULTS_DIR = ROOT / 'results/bert'

LABELS = {"topic": ["billing", "technical", "sales", "other"], "urgent": [0, 1], "sentiment": [0, 1, 2]}


def settings():
    return {
        'base_model': BERT_MODEL,
        'revision': BERT_REVISION,
        'model_dir': BERT_OUTPUT_DIR,
        'train_file': TRAIN_FILE,
        'validation_file': VALIDATION_FILE,
        'test_file': TEST_FILE,
        'epochs': BERT_EPOCHS,
        'batch_size': BERT_BATCH_SIZE,
        'max_length': BERT_MAX_LENGTH,
        'lr': BERT_LEARNING_RATE,
        'seed': BERT_SEED,
        'device': BERT_DEVICE,
        'results_dir': BERT_RESULTS_DIR,
    }


def device_for(config):
    if config["device"] != "auto":
        return torch.device(config["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class BatchGenerator:
    """Выдаёт строки и токенизированные тензоры по одному батчу."""

    def __init__(self, rows, tokenizer, config, device):
        self.rows = rows
        self.tokenizer = tokenizer
        self.batch_size = config["batch_size"]
        self.max_length = config["max_length"]
        self.device = device

    def __iter__(self):
        for start in range(0, len(self.rows), self.batch_size):
            subset = self.rows[start:start + self.batch_size]
            encoded = self.tokenizer(
                [row["text"] for row in subset],
                padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            yield subset, {key: value.to(self.device) for key, value in encoded.items()}


def metrics(truth, predicted, classes):
    names = [str(label) for label in classes]
    report = classification_report(
        truth, predicted, labels=list(range(len(classes))),
        target_names=names, output_dict=True, zero_division=0,
    )
    return {
        "count": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in names
        },
    }


def plot_history(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for task, reports in history.items():
        epochs = [report["epoch"] for report in reports]
        axes[0].plot(epochs, [report["train_loss"] for report in reports], marker="o", label=task)
        axes[1].plot(epochs, [report["macro_f1"] for report in reports], marker="o", label=task)
    axes[0].set(title="Training loss (mean per example)", ylabel="Loss")
    axes[1].set(title="Validation macro-F1", ylabel="F1", ylim=(0, 1))
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def main():
    config = settings()

    if config["model_dir"].exists():
        raise ValueError("Каталог весов уже существует. Укажи новый BERT_OUTPUT_DIR в начале bert/train.py.")

    train, validation = read_rows(config["train_file"]), read_rows(config["validation_file"])

    for field in ("id", "group_id"):
        if {r[field] for r in train} & {r[field] for r in validation}:
            raise ValueError(f"Пересечение train/validation по {field}")


    set_seed(config["seed"])
    
    device = device_for(config)

    print(f"Устройство: {device}; train={len(train)}, validation={len(validation)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])

    config["model_dir"].mkdir(parents=True)
    history = {}

    for task, labels in LABELS.items():
        set_seed(config["seed"])

        model = AutoModelForSequenceClassification.from_pretrained(
            config["base_model"], revision=config["revision"], num_labels=len(labels),
            ignore_mismatched_sizes=True,
            id2label={i: str(label) for i, label in enumerate(labels)},
            label2id={str(label): i for i, label in enumerate(labels)},
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

        best_f1 = -1.0

        history[task] = []
        for epoch in range(config["epochs"]):
            order = torch.randperm(len(train)).tolist()
            shuffled = [train[i] for i in order]
            model.train()
            total_loss = 0.0

            for subset, inputs in BatchGenerator(shuffled, tokenizer, config, device):
                target = torch.tensor([labels.index(r["labels"][task]) for r in subset], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(**inputs, labels=target).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(subset)

            model.eval()
            truth, predicted = [], []
            with torch.inference_mode():
                for subset, inputs in BatchGenerator(validation, tokenizer, config, device):
                    predicted.extend(model(**inputs).logits.argmax(-1).cpu().tolist())
                    truth.extend(labels.index(r["labels"][task]) for r in subset)

            report = metrics(truth, predicted, labels)
            report.update(epoch=epoch + 1, train_loss=total_loss / len(train))
            history[task].append(report)

            print(f"{task}, эпоха {epoch+1}: loss={report['train_loss']:.4f}, val macro-F1={report['macro_f1']:.4f}", flush=True)

            if report["macro_f1"] > best_f1:
                best_f1 = report["macro_f1"]
                destination = config["model_dir"] / task
                model.save_pretrained(destination, safe_serialization=True)
                tokenizer.save_pretrained(destination)

        del optimizer, model

    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    metadata.update(labels=LABELS, history=history)
    (config["model_dir"] / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    plot_history(history, config["model_dir"])


if __name__ == "__main__":
    main()
