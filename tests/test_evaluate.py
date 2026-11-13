"""
What:     Tests scripts/evaluate.py: field outcomes, row alignment, failed documents, calibration, the report, and the private-data guard.
In:       Mocked fixture answers (tests/helpers.py) through a fake LLM client; a temporary run folder. No network.
Out:      pytest pass/fail.
Fails if: The scoring rules change. They define the accuracy numbers in the README, so changes must be deliberate.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.assemble import build_invoice
from app.llm import LLMReply
from app.models import ExpectedInvoice, LLMInvoiceExtraction
from scripts.evaluate import (
    FieldScore,
    by_field,
    calibration,
    evaluate,
    fixtures,
    main,
    overall,
    render,
    score_invoice,
    tally,
)
from tests.helpers import FIXTURES, as_json, expected, pages_of, wire_01, wire_03, wire_05

TODAY = date(2026, 11, 13)


def invoice_of(stem: str, wire: dict):
    return build_invoice(LLMInvoiceExtraction.model_validate(wire), list(pages_of(stem))).invoice


def outcomes(scores, prefix: str) -> dict[str, str]:
    return {s.path: s.outcome for s in scores if s.path.startswith(prefix)}


# --- Outcomes ------------------------------------------------------------------------------------


def test_perfect_answer_scores_every_field_correct():
    scores = score_invoice("01", invoice_of("01_gst_classic", wire_01()), expected("01_gst_classic"))
    assert all(s.correct for s in scores)
    assert {s.outcome for s in scores} == {"correct", "absent"}  # taxable_value and tax_total are correctly absent


def test_each_outcome_kind():
    wire = wire_01()
    wire["invoice_number"]["value_text"] = "SOS"  # printed on the line, but not the whole number -> wrong
    wire["due_date"] = {"value_text": None, "page": None, "line_start": None, "line_end": None, "not_found_reason": "x"}  # -> missed
    wire["tax_total"] = {**wire["subtotal"]}  # a value where nothing is printed -> spurious
    scores = {s.path: s for s in score_invoice("01", invoice_of("01_gst_classic", wire), expected("01_gst_classic"))}
    assert scores["invoice_number"].outcome == "wrong" and scores["invoice_number"].actual == "SOS"
    assert scores["due_date"].outcome == "missed" and scores["due_date"].confidence is None
    assert scores["tax_total"].outcome == "spurious" and scores["tax_total"].actual == Decimal("67367.00")
    assert scores["taxable_value"].outcome == "absent"


def test_missing_and_extra_rows():
    wire = wire_01()
    wire["line_items"] = wire["line_items"][:4]
    scores = score_invoice("01", invoice_of("01_gst_classic", wire), expected("01_gst_classic"))
    assert outcomes(scores, "line_items.count") == {"line_items.count": "wrong"}
    assert set(outcomes(scores, "line_items.4.").values()) == {"missed"}

    wire = wire_01()
    wire["line_items"].append(wire["line_items"][0])
    scores = score_invoice("01", invoice_of("01_gst_classic", wire), expected("01_gst_classic"))
    assert set(outcomes(scores, "line_items.5.").values()) == {"spurious"}


def test_adjustment_label_is_graded_only_when_ground_truth_has_one():
    exp = expected("03_multipage_roundoff")
    inv = invoice_of("03_multipage_roundoff", wire_03())
    assert "adjustments.0.label" in {s.path for s in score_invoice("03", inv, exp)}
    no_labels = ExpectedInvoice.model_validate(
        exp.model_dump(mode="json") | {"adjustments": [{"kind": a.kind, "amount": str(a.amount)} for a in exp.adjustments]}
    )
    assert "adjustments.0.label" not in {s.path for s in score_invoice("03", inv, no_labels)}


def test_failed_document_counts_as_wrong_everywhere():
    scores = score_invoice("05", None, expected("05_eu_vat_defective"), flags=[])
    assert not any(s.correct and s.outcome != "absent" for s in scores)
    assert next(s for s in scores if s.field == "flags").outcome == "wrong"
    assert all(s.outcome in ("missed", "absent", "wrong") for s in scores)


def test_flags_must_match_exactly():
    from app.validate import validate

    inv = invoice_of("05_eu_vat_defective", wire_05())
    flags = validate(inv, TODAY)
    [flag_score] = [s for s in score_invoice("05", inv, expected("05_eu_vat_defective"), flags) if s.field == "flags"]
    assert flag_score.correct
    [flag_score] = [s for s in score_invoice("05", inv, expected("05_eu_vat_defective"), flags[:1]) if s.field == "flags"]
    assert not flag_score.correct


# --- Aggregation ---------------------------------------------------------------------------------


def fs(outcome: str, field: str = "subtotal", confidence: float | None = 1.0) -> FieldScore:
    actual = "value" if confidence is not None else None  # a confidence only exists for a returned value
    return FieldScore("d", field, field, None, actual, outcome, confidence)


def test_tally_and_accuracy():
    t = tally("x", [fs("correct"), fs("absent"), fs("wrong"), fs("missed"), fs("spurious")])
    assert (t.n, t.correct, t.absent, t.wrong, t.missed, t.spurious) == (5, 1, 1, 1, 1, 1)
    assert t.accuracy == 0.4


def test_overall_excludes_page_and_flags():
    scores = [fs("correct"), fs("wrong", "page"), fs("wrong", "flags", None)]
    assert overall(scores).n == 1


def test_by_field_follows_display_order_and_skips_empty_fields():
    scores = [fs("correct", "grand_total"), fs("correct", "invoice_number")]
    assert [t.label for t in by_field(scores)] == ["invoice_number", "grand_total"]


def test_calibration_bands():
    scores = [fs("correct", confidence=1.0), fs("wrong", confidence=0.96), fs("correct", confidence=0.9),
              fs("correct", confidence=0.5), FieldScore("d", "x", "due_date", None, None, "absent", None)]
    bands = {t.label: (t.n, t.accuracy) for t in calibration(scores)}
    assert bands["0.95 – 1.00"] == (2, 0.5)
    assert bands["0.80 – 0.95"] == (1, 1.0)
    assert bands["below 0.80"] == (1, 1.0)
    assert bands["no value returned"] == (1, 1.0)


# --- Full run (fake LLM, no network) ---------------------------------------------------------------

ANSWERS = {"Sharma Office": wire_01, "Ganga Electricals": wire_03, "Müller": wire_05}


class FakeClient:
    def complete(self, *, system, messages):
        wire = next(w for key, w in ANSWERS.items() if key in messages[0]["content"])
        return LLMReply(text=as_json(wire()), content=[], model="mock", provider="fake")


def test_evaluate_writes_a_report_and_results(tmp_path):
    items = [i for i in fixtures(None, False) if i[0] in ("01_gst_classic", "03_multipage_roundoff", "05_eu_vat_defective")]
    runs, report = evaluate(items, FakeClient, tmp_path, TODAY, {"Run": "test"}, log=lambda _: None)
    assert [r.stem for r in runs] == ["01_gst_classic", "03_multipage_roundoff", "05_eu_vat_defective"]
    assert all(s.correct for r in runs for s in r.scores)
    assert "**" in report and "100.0%" in report and "| 05_eu_vat_defective | done |" in report
    assert "Answered by:** fake/mock" in report and "## Every incorrect field\n\nNone." in report
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == [f"{s}.json" for s in ("01_gst_classic", "03_multipage_roundoff", "05_eu_vat_defective")]
    assert (tmp_path / "report.md").read_text() == report and (tmp_path / "scores.json").is_file()


def test_failed_document_appears_in_the_report(tmp_path):
    class Down:
        def complete(self, **_):
            from app.llm import LLMAPIError

            raise LLMAPIError("Gemini free-tier daily quota is used up")

    items = [i for i in fixtures(["01_gst_classic"], False)]
    runs, report = evaluate(items, Down, tmp_path, TODAY, {}, log=lambda _: None)
    assert runs[0].result.status.value == "failed"
    assert "| 01_gst_classic | failed (Gemini free-tier daily quota is used up) |" in report
    assert overall(runs[0].scores).accuracy < 0.5


def test_render_lists_every_incorrect_field():
    wire = wire_01()
    wire["due_date"] = {"value_text": None, "page": None, "line_start": None, "line_end": None, "not_found_reason": "x"}
    scores = score_invoice("01", invoice_of("01_gst_classic", wire), expected("01_gst_classic"))
    from scripts.evaluate import DocRun
    from app.pipeline import new_job
    from datetime import UTC, datetime

    run = DocRun("01", new_job(b"%PDF", "01.pdf", datetime.now(UTC)), scores, 1.0)
    assert "| 01 | due_date | 2025-11-20 | ∅ | missed | – |" in render([run], {})


# --- Fixture discovery and the private-data guard ---------------------------------------------------


def test_fixture_discovery_uses_the_owners_files():
    stems = [stem for stem, _, _ in fixtures(None, False)]
    assert stems == sorted(p.stem for p in (FIXTURES / "expected").glob("*.json") if not p.name.startswith("_"))
    assert all(pdf.is_file() for _, pdf, _ in fixtures(None, False))
    assert [s for s, _, _ in fixtures(["03_multipage_roundoff"], False)] == ["03_multipage_roundoff"]


def test_private_fixtures_are_never_sent_to_gemini(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert main(["--include-private"]) == 2
    assert "Refusing to send tests/fixtures/private through Gemini" in capsys.readouterr().err
