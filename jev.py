"""Минимальный клиент Jev; сторонние зависимости не нужны."""
import argparse
import hashlib
import http.client
import json
import math
import os
import socket
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from telemetry import Telemetry

ROOT = Path(__file__).resolve().parent

JEV_MODEL = 'jev-latest'
INPUT_FILE = ROOT / 'data/authored-v2-en/test.jsonl'
OUTPUT_FILE = ROOT / 'results/jev_en.jsonl'
TIMEOUT_SECONDS = 60.0
INPUT_USD_PER_MILLION = 0.042
OUTPUT_USD_PER_MILLION = 0.0
PRICE_SOURCE = "https://typesafe.ai/blog/introducing-system-one-models-and-jev"


def usage_cost(response, input_rate, output_rate):
    usage = response.get("usage") or {}
    valid = lambda x: isinstance(x, int) and not isinstance(x, bool) and x >= 0
    counts = {name: usage.get(name) if valid(usage.get(name)) else None
              for name in ("input_tokens", "output_tokens")}
    cost = 0.0
    for name, rate in (("input_tokens", input_rate), ("output_tokens", output_rate)):
        if rate and counts[name] is None:
            return counts, None
        cost += (counts[name] or 0) * rate / 1e6
    return counts, cost


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
        "instructions": "Choose the main topic of the request. If several topics are mentioned, follow the customer's primary request.",
        "criteria": {
            "billing": "Payments, refunds, charges, invoices and settlement of existing financial obligations",
            "technical": "Application faults, integrations, account access and product configuration",
            "sales": "Questions about pricing, plans, licensing or a new purchase",
            "other": "All other requests",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "The customer explicitly requests immediate assistance or describes an ongoing interruption of work or loss of sales. Dissatisfaction alone does not imply urgency.",
    },
    "sentiment": {
        "type": "score",
        "instructions": "Assess the emotional tone explicitly expressed in the message using the scale.",
        "criteria": [
            "Negative: dissatisfaction or irritation",
            "Neutral: facts or a question without expressed emotion",
            "Positive: gratitude or satisfaction",
        ],
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


class JevRequestError(RuntimeError):
    """Безопасная диагностика без API-ключа или тела запроса."""
    def __init__(self, message, kind, http_status=None, retry_after=None):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status
        self.retry_after = retry_after


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
        hints = {400: "Некорректный запрос", 401: "Некорректный API-ключ", 402: "Недостаточно средств",
                 403: "Доступ запрещён", 404: "Endpoint или модель не найдены", 422: "Ошибка схемы запроса",
                 429: "Превышен лимит запросов"}
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        exc.close()
        hint = hints.get(exc.code, "Ошибка сервиса" if exc.code >= 500 else "Запрос отклонён")
        raise JevRequestError(f"Jev HTTP {exc.code}: {hint}. Автоматического повтора нет.",
                              "http", exc.code, retry_after) from None
    except urllib.error.URLError as exc:
        kind = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "network"
        raise JevRequestError("Таймаут запроса Jev" if kind == "timeout" else "Сетевая ошибка Jev (DNS, TLS или соединение)", kind) from None
    except TimeoutError:
        raise JevRequestError("Таймаут запроса Jev", "timeout") from None
    except (http.client.HTTPException, OSError):
        raise JevRequestError("Соединение Jev оборвано или HTTP-ответ повреждён", "connection") from None
    except (json.JSONDecodeError, UnicodeError):
        raise JevRequestError("Jev вернул некорректный JSON", "invalid_json") from None
    if (not isinstance(result, dict) or not isinstance(result.get("answers"), dict)
            or not set(payload["questions"]).issubset(result["answers"])
            or any(not isinstance(result["answers"][name], dict) for name in payload["questions"])
            or (result.get("usage") is not None and not isinstance(result["usage"], dict))):
        raise JevRequestError("Jev вернул неожиданную структуру ответа", "invalid_response")
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
    run.add_argument("--input-usd-per-million", type=float, default=INPUT_USD_PER_MILLION)
    run.add_argument("--output-usd-per-million", type=float, default=OUTPUT_USD_PER_MILLION)
    args = parser.parse_args(sys.argv[1:] or ["run"])

    if any(not math.isfinite(rate) or rate < 0 for rate in (args.input_usd_per_million, args.output_usd_per_million)):
        parser.error("Тарифы должны быть конечными неотрицательными числами")
    rows = read_rows(args.input)

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout должен быть положительным")
    key = os.environ.get("TYPESAFE_API_KEY")

    if not key:
        raise ValueError("Заполни TYPESAFE_API_KEY в .env")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x") as handle:
        telemetry = Telemetry("jev", args.input, args.output.parent, {
            "output": str(args.output), "requested_model": args.model, "timeout_seconds": args.timeout,
            "batch_size": 1, "concurrency": 1, "expected_examples": len(rows),
            "latency_definition": "HTTP including network and JSON decoding; no retries or warmup",
            "pricing": {"currency": "USD", "input_per_million": args.input_usd_per_million,
                        "output_per_million": args.output_usd_per_million, "source": PRICE_SOURCE,
                        "verified_date": "2026-10-04", "kind": "estimate_from_usage_not_invoice"}})
        completed = False
        try:
            for row in rows:
                payload = request_payload(row["text"], args.model)
                started = time.perf_counter()
                try:
                    response, elapsed = call_jev(payload, key, args.timeout)
                except Exception as exc:
                    telemetry.record(id=row["id"], phase="measured", status="error",
                                     latency_seconds=time.perf_counter() - started,
                                     error_type=type(exc).__name__, error=str(exc), estimated_cost_usd=None,
                                     error_kind=getattr(exc, "kind", None), http_status=getattr(exc, "http_status", None),
                                     retry_after=getattr(exc, "retry_after", None))
                    raise
                usage, cost = usage_cost(response, args.input_usd_per_million, args.output_usd_per_million)
                record = {"id": row["id"], "requested_model": args.model, "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "latency_seconds": elapsed, "response": response,
                          "usage": usage, "estimated_cost_usd": cost}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                telemetry.record(**record, phase="measured", status="ok", input_characters=len(row["text"]))
                print(json.dumps(record, ensure_ascii=False, indent=2))
            completed = True
        finally:
            costs = [r.get("estimated_cost_usd") for r in telemetry.records]
            telemetry.finish(completed, {
                "estimated_cost_usd_known": sum(c for c in costs if c is not None),
                "cost_unknown_attempts": sum(c is None for c in costs),
                "estimated_total_cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
                "usage_known": {name: sum((r.get("usage") or {}).get(name) or 0 for r in telemetry.records)
                                for name in ("input_tokens", "output_tokens")}})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
