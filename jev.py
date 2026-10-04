"""Минимальный клиент Jev; сторонние зависимости не нужны."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent

JEV_MODEL = 'jev-latest'
INPUT_FILE = ROOT / 'data/authored-v2/test.jsonl'
OUTPUT_FILE = ROOT / 'results/jev.jsonl'
TIMEOUT_SECONDS = 60.0


def load_env():
    """Загрузить API-ключ Jev из .env."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Некорректная строка .env:{number}; ожидается KEY=value")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key != "TYPESAFE_API_KEY":
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "Выбери основную тему обращения. Если тем несколько, выбери тему главной просьбы клиента.",
        "criteria": {
            "billing": "Оплата, возврат денег, списания, счета",
            "technical": "Ошибка приложения, интеграции или доступ к аккаунту",
            "sales": "Вопрос о цене, тарифах или покупке",
            "other": "Все остальные обращения",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Клиент явно просит немедленной помощи или описывает текущую остановку работы либо потерю продаж. Недовольство само по себе не означает срочность.",
    },
    "sentiment": {
        "type": "score",
        "instructions": "Оцени выраженный в сообщении эмоциональный тон по шкале.",
        "criteria": ["Негативный: недовольство или раздражение", "Нейтральный: факты или вопрос без выраженной эмоции", "Позитивный: благодарность или удовлетворение"],
    },
}


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Каждая запись должна содержать непустой text")
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Каждая запись должна иметь уникальный строковый id")
        ids.add(row["id"])
    return rows


def request_payload(text, model):
    return {"state": text, "model": model, "questions": QUESTIONS}


def call_jev(payload, key, timeout):
    request = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Jev вернул HTTP {exc.code}. Проверь ключ, баланс и формат запроса.") from None
    return result, time.perf_counter() - started


def main():
    load_env()

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Обработать все записи через API")
    run.add_argument("--input", type=Path, default=INPUT_FILE)
    run.add_argument("--output", type=Path, default=OUTPUT_FILE)
    run.add_argument("--model", default=JEV_MODEL)
    run.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args(sys.argv[1:] or ["run"])

    rows = read_rows(args.input)

    if args.timeout <= 0:
        raise ValueError("timeout должен быть положительным")
    key = os.environ.get("TYPESAFE_API_KEY")

    if not key:
        raise ValueError("Заполни TYPESAFE_API_KEY в .env")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x") as handle:
        for row in rows:
            payload = request_payload(row["text"], args.model)
            response, elapsed = call_jev(payload, key, args.timeout)
            record = {"id": row["id"], "requested_model": args.model, "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "latency_seconds": elapsed, "response": response}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
