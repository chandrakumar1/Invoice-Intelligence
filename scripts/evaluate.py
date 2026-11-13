"""
What:     Accuracy harness: runs every fixture that has hand-written ground truth through the real pipeline and scores each field.
In:       tests/fixtures/pdfs + tests/fixtures/expected (and, with --include-private, tests/fixtures/private); LLM_PROVIDER / LLM_MODEL.
Out:      Markdown report on stdout, plus a run folder data/eval/<timestamp>/ with each ExtractionResult, scores.json and report.md.
Fails if: The provider's quota runs out mid-run. Finished documents are kept; the failed one is reported as failed, and all
          its fields count as wrong, so a failure can never improve the score.

Each field gets one outcome:
  correct   both present and equal              absent    both null (correctly not extracted)
  wrong     both present, different             missed    expected a value, got none
  spurious  expected none (not printed), got a value: the hallucination case
Accuracy = (correct + absent) / all. The comparison rules here are the only definition of "correct" in the repo;
tests/helpers.compare delegates to score_invoice.

Usage: .venv/bin/python scripts/evaluate.py [--only STEM ...] [--include-private] [--out DIR] [--today YYYY-MM-DD]
"""

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import PROMPT_VERSION, LLMClient, client_from_env  # noqa: E402
from app.models import ExpectedInvoice, Extracted, ExtractionResult, Invoice, JobStatus, LineItem, ValidationFlag  # noqa: E402
from app.pipeline import new_job, process  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
CORRECT = ("correct", "absent")
CELLS = ("description", "hsn_sac", "quantity", "unit_price", "amount")


@dataclass(frozen=True)
class FieldScore:
    doc: str
    path: str  # exact location, e.g. "line_items.3.amount"
    field: str  # aggregation key, e.g. "line_items.amount"
    expected: Any
    actual: Any
    outcome: str  # correct | absent | wrong | missed | spurious
    confidence: float | None = None  # confidence of the returned value; None if nothing was returned

    @property
    def correct(self) -> bool:
        return self.outcome in CORRECT


# --- Scoring --------------------------------------------------------------------------------------


def _outcome(actual: Any, expected: Any) -> str:
    if actual is None and expected is None:
        return "absent"
    if expected is None:
        return "spurious"
    if actual is None:
        return "missed"
    return "correct" if actual == expected else "wrong"  # Decimal("9") == Decimal("9.00"): compared by value


def _ws(s: str | None) -> str | None:
    return None if s is None else re.sub(r"\s+", " ", s).strip()


def _loose(s: str | None) -> str | None:
    """Addresses: runs of commas and whitespace count as one separator (owner's decision)."""
    return None if s is None else re.sub(r"[,\s]+", " ", s).strip()


def score_invoice(doc: str, inv: Invoice | None, exp: ExpectedInvoice, flags: list[ValidationFlag] | None = None) -> list[FieldScore]:
    """Score one document. `inv=None` (extraction failed) scores every expected value as missed.
    `flags=None` skips the flags comparison (used when only extraction is being compared)."""
    scores: list[FieldScore] = []

    def add(path: str, field: str, actual: Any, expected: Any, confidence: float | None = None) -> None:
        scores.append(FieldScore(doc, path, field, expected, actual, _outcome(actual, expected), confidence if actual is not None else None))

    def ext(value: Extracted | None, normalize: Callable = lambda v: v) -> tuple[Any, float | None]:
        if value is None or value.value is None:
            return None, None
        return normalize(value.value), value.source.confidence

    for name in ("invoice_number", "invoice_date", "due_date", "currency", "subtotal", "taxable_value", "tax_total", "grand_total"):
        actual, conf = ext(getattr(inv, name) if inv else None, _ws if name in ("invoice_number", "currency") else (lambda v: v))
        add(name, name, actual, getattr(exp, name), conf)

    for side in ("seller", "buyer"):
        got = getattr(inv, side) if inv else None
        want = getattr(exp, side)
        actual, conf = ext(got.name if got else None, _ws)
        add(f"{side}.name", "party.name", actual, _ws(want.name), conf)
        actual, conf = ext(got.address if got else None, _loose)
        add(f"{side}.address", "party.address", actual, _loose(want.address), conf)
        tax_id, conf = ext(got.tax_id if got else None)
        add(f"{side}.tax_id", "party.tax_id", tax_id.raw.strip() if tax_id else None, want.tax_id, conf)
        add(f"{side}.tax_scheme", "party.tax_scheme", tax_id.scheme if tax_id else None, want.tax_scheme, conf)

    rows = inv.line_items if inv else []
    add("line_items.count", "line_items.count", len(rows) if inv else None, len(exp.line_items))
    for i in range(max(len(rows), len(exp.line_items))):
        got = rows[i] if i < len(rows) else None
        want = exp.line_items[i] if i < len(exp.line_items) else None
        for cell in CELLS:
            actual = getattr(got, cell) if got else None
            wanted = getattr(want, cell) if want else None
            if isinstance(actual, str):
                actual, wanted = _ws(actual), _ws(wanted)
            add(f"line_items.{i}.{cell}", f"line_items.{cell}", actual, wanted, got.source.confidence if got else None)

    taxes = inv.tax_lines if inv else []
    add("tax_lines.count", "tax_lines.count", len(taxes) if inv else None, len(exp.tax_lines))
    for i in range(max(len(taxes), len(exp.tax_lines))):
        got = taxes[i] if i < len(taxes) else None
        want = exp.tax_lines[i] if i < len(exp.tax_lines) else None
        actual, conf = ext(got.label if got else None, _ws)
        add(f"tax_lines.{i}.label", "tax_lines.label", actual, _ws(want.label) if want else None, conf)
        for part in ("rate", "amount"):
            actual, conf = ext(getattr(got, part) if got else None)
            add(f"tax_lines.{i}.{part}", f"tax_lines.{part}", actual, getattr(want, part) if want else None, conf)

    adjs = inv.adjustments if inv else []
    add("adjustments.count", "adjustments.count", len(adjs) if inv else None, len(exp.adjustments))
    for i in range(max(len(adjs), len(exp.adjustments))):
        got = adjs[i] if i < len(adjs) else None
        want = exp.adjustments[i] if i < len(exp.adjustments) else None
        amount, conf = ext(got.amount if got else None)
        add(f"adjustments.{i}.kind", "adjustments.kind", got.kind if got else None, want.kind if want else None, conf)
        add(f"adjustments.{i}.amount", "adjustments.amount", amount, want.amount if want else None, conf)
        if want is not None and want.label is not None:  # labels are optional in ground truth; graded only when given
            label, conf = ext(got.label if got else None, _ws)
            add(f"adjustments.{i}.label", "adjustments.label", label, _ws(want.label), conf)

    for path, page in exp.expected_pages.items():
        add(f"page of {path}", "page", page_of(inv, path) if inv else None, page)

    if flags is not None:
        wanted = sorted(c.value for c in exp.expected_flags)
        got_codes = sorted({f.code.value for f in flags})
        outcome = "correct" if got_codes == wanted else "wrong"
        scores.append(FieldScore(doc, "flags", "flags", wanted, got_codes, outcome if inv else "wrong"))
    return scores


def page_of(inv: Invoice, path: str) -> int | None:
    node: object = inv
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)] if int(part) < len(node) else None
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    if isinstance(node, Extracted):
        return node.source.page if node.source else None
    if isinstance(node, LineItem):
        return node.source.page
    amount = getattr(node, "amount", None)  # TaxLine / Adjustment: where the amount is printed
    return amount.source.page if isinstance(amount, Extracted) and amount.source else None


# --- Aggregation --------------------------------------------------------------------------------------

FIELD_ORDER = [
    "invoice_number", "invoice_date", "due_date", "currency",
    "party.name", "party.address", "party.tax_id", "party.tax_scheme",
    "line_items.count", *(f"line_items.{c}" for c in CELLS),
    "subtotal", "taxable_value",
    "tax_lines.count", "tax_lines.label", "tax_lines.rate", "tax_lines.amount", "tax_total",
    "adjustments.count", "adjustments.kind", "adjustments.amount", "adjustments.label",
    "grand_total", "page", "flags",
]
CONFIDENCE_BANDS = [("0.95 – 1.00", 0.95, 1.01), ("0.80 – 0.95", 0.80, 0.95), ("below 0.80", 0.0, 0.80)]


@dataclass(frozen=True)
class Tally:
    label: str
    n: int
    correct: int
    absent: int
    wrong: int
    missed: int
    spurious: int

    @property
    def accuracy(self) -> float:
        return (self.correct + self.absent) / self.n if self.n else 0.0


def tally(label: str, scores: Iterable[FieldScore]) -> Tally:
    s = list(scores)
    count = lambda o: sum(x.outcome == o for x in s)  # noqa: E731
    return Tally(label, len(s), count("correct"), count("absent"), count("wrong"), count("missed"), count("spurious"))


def by_field(scores: list[FieldScore]) -> list[Tally]:
    known = [tally(f, [s for s in scores if s.field == f]) for f in FIELD_ORDER]
    return [t for t in known if t.n]


def overall(scores: list[FieldScore]) -> Tally:
    """Every extracted field; excludes page attribution and flags, which are reported separately."""
    return tally("all extracted fields", [s for s in scores if s.field not in ("page", "flags")])


def calibration(scores: list[FieldScore]) -> list[Tally]:
    """Accuracy of returned values by their confidence score. Counts and page/flag rows have no confidence."""
    returned = [s for s in scores if s.confidence is not None]
    bands = [tally(label, [s for s in returned if lo <= s.confidence < hi]) for label, lo, hi in CONFIDENCE_BANDS]
    nothing = [s for s in scores if s.actual is None and s.field not in ("flags",)]
    return [*bands, tally("no value returned", nothing)]


# --- Report ---------------------------------------------------------------------------------------------


def _pct(t: Tally) -> str:
    return f"{t.accuracy:.1%}" if t.n else "–"


def _fmt(v: Any) -> str:
    if v is None:
        return "∅"
    if isinstance(v, (Decimal, date)):
        return str(v)
    if isinstance(v, list):
        return ", ".join(map(str, v)) or "(none)"
    return str(getattr(v, "value", v))


@dataclass
class DocRun:
    stem: str
    result: ExtractionResult
    scores: list[FieldScore]
    seconds: float


def render(runs: list[DocRun], meta: dict[str, str]) -> str:
    scores = [s for r in runs for s in r.scores]
    out = ["# Accuracy report", ""]
    out += [f"- **{k}:** {v}" for k, v in meta.items()]
    total = overall(scores)
    out += ["", f"**{total.correct + total.absent} / {total.n} extracted fields correct ({_pct(total)})**, "
            f"{total.wrong} wrong, {total.missed} missed, {total.spurious} spurious.", ""]

    out += ["## Per document", "", "| Document | Status | Pages | Fields correct | Flags match | Time |", "|---|---|---|---|---|---|"]
    for r in runs:
        t = overall(r.scores)
        sources = ", ".join(p.text_source.replace("_", " ") for p in r.result.pages) or "–"
        flag = next((s for s in r.scores if s.field == "flags"), None)
        flags = ("yes" if flag.correct else "no") if flag else "–"
        status = r.result.status.value + (f" ({r.result.error})" if r.result.error else "")
        out.append(f"| {r.stem} | {status} | {sources} | {t.correct + t.absent}/{t.n} ({_pct(t)}) | {flags} | {r.seconds:.0f} s |")

    out += ["", "## Per field", "", "| Field | n | Accuracy | Correct | Correctly absent | Wrong | Missed | Spurious |", "|---|---|---|---|---|---|---|---|"]
    for t in by_field(scores):
        out.append(f"| {t.label} | {t.n} | {_pct(t)} | {t.correct} | {t.absent} | {t.wrong} | {t.missed} | {t.spurious} |")

    out += ["", "## Confidence calibration", "", "Are high-confidence values more often right?", "",
            "| Confidence | Values | Accuracy |", "|---|---|---|"]
    for t in calibration(scores):
        out.append(f"| {t.label} | {t.n} | {_pct(t)} |")

    misses = [s for s in scores if not s.correct]
    out += ["", "## Every incorrect field", ""]
    if not misses:
        out.append("None.")
    else:
        out += ["| Document | Field | Expected | Got | Outcome | Confidence |", "|---|---|---|---|---|---|"]
        for s in misses:
            conf = f"{s.confidence:.2f}" if s.confidence is not None else "–"
            out.append(f"| {s.doc} | {s.path} | {_fmt(s.expected)} | {_fmt(s.actual)} | {s.outcome} | {conf} |")
    return "\n".join(out) + "\n"


# --- Running -----------------------------------------------------------------------------------------------


def fixtures(only: list[str] | None, include_private: bool) -> list[tuple[str, Path, ExpectedInvoice]]:
    found = []
    folders = [(FIXTURES / "expected", FIXTURES / "pdfs")]
    if include_private:
        folders.append((FIXTURES / "private", FIXTURES / "private"))
    for expected_dir, pdf_dir in folders:
        for path in sorted(expected_dir.glob("*.json")):
            if path.name.startswith("_") or (only and path.stem not in only):
                continue
            exp = ExpectedInvoice.model_validate_json(path.read_text())
            found.append((path.stem, pdf_dir / exp.fixture, exp))
    return found


def run_document(stem: str, pdf: Path, exp: ExpectedInvoice, llm_factory: Callable[[], LLMClient], today: date) -> DocRun:
    data = pdf.read_bytes()
    started = time.monotonic()
    result = process(new_job(data, pdf.name, datetime.now(UTC)), data, llm_factory=llm_factory, today=today)
    seconds = time.monotonic() - started
    done = result.status is JobStatus.DONE
    return DocRun(stem, result, score_invoice(stem, result.invoice if done else None, exp, result.flags), seconds)


def evaluate(
    items: list[tuple[str, Path, ExpectedInvoice]],
    llm_factory: Callable[[], LLMClient],
    out_dir: Path,
    today: date,
    meta: dict[str, str],
    log: Callable[[str], None] = print,
) -> tuple[list[DocRun], str]:
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    runs = []
    for i, (stem, pdf, exp) in enumerate(items, 1):
        log(f"[{i}/{len(items)}] {stem} ...")
        run = run_document(stem, pdf, exp, llm_factory, today)
        (out_dir / "results" / f"{stem}.json").write_text(run.result.model_dump_json(indent=2))
        t = overall(run.scores)
        log(f"        {run.result.status.value}: {t.correct + t.absent}/{t.n} fields correct, {run.seconds:.0f} s")
        runs.append(run)
    models = sorted({f"{r.result.provider}/{r.result.model}" for r in runs if r.result.model})
    report = render(runs, {**meta, "Answered by": ", ".join(models) or "– (no document completed)"})
    (out_dir / "report.md").write_text(report)
    (out_dir / "scores.json").write_text(json.dumps([asdict(s) for s in (s for r in runs for s in r.scores)], default=str, indent=1))
    return runs, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].removeprefix("What:").strip())
    parser.add_argument("--only", nargs="+", metavar="STEM", help="run only these fixtures, e.g. 01_gst_classic")
    parser.add_argument("--include-private", action="store_true", help="also run tests/fixtures/private (real invoices)")
    parser.add_argument("--out", type=Path, help="run folder (default: data/eval/<timestamp>)")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(), help="date used by the 'not in the future' check")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    provider = (os.environ.get("LLM_PROVIDER") or "gemini").lower()
    if args.include_private and provider == "gemini":
        print(
            "Refusing to send tests/fixtures/private through Gemini: on the free tier Google may use the data and human\n"
            "reviewers may read it. Use LLM_PROVIDER=anthropic, or a billed Google Cloud project with a different provider setup.",
            file=sys.stderr,
        )
        return 2

    items = fixtures(args.only, args.include_private)
    if not items:
        print("No fixtures with ground truth matched.", file=sys.stderr)
        return 1
    started = datetime.now(UTC)
    out_dir = args.out or ROOT / "data" / "eval" / started.strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "Run": started.strftime("%Y-%m-%d %H:%M UTC"),
        "Documents": str(len(items)),
        "Prompt": PROMPT_VERSION,
        "Validation date": args.today.isoformat(),
    }
    _, report = evaluate(items, client_from_env, out_dir, args.today, meta)
    print()
    print(report)
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
