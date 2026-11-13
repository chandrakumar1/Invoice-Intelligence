<!--
What:     Explains the fixture folders and how to hand-write ground truth (also documents the JSON files, which can't hold comments).
In:       Read by the owner before writing expected/*.json.
Out:      Expected files that load into app.models.ExpectedInvoice.
Fails if: An expected file is written from generator data or LLM output instead of by reading the PDF. Then it tests nothing.
-->

# Test fixtures

| Folder | Contents | In git? |
|---|---|---|
| `pdfs/` | Synthetic invoices from `scripts/generate_fixtures.py` (byte-identical on regeneration) | yes |
| `expected/` | **Hand-written** ground truth, one `<pdf-stem>.json` per PDF | yes |
| `private/` | Real-world invoices plus their expected JSON (PII) | **no** (gitignored) |

## The synthetic set

| File | What it exercises |
|---|---|
| `01_gst_classic.pdf` | Standard Indian B2B tax invoice, intra-state CGST + SGST, Indian digit grouping (`1,23,456.00`), DD/MM/YYYY |
| `02_igst_totals_top.pdf` | Inter-state IGST, totals box printed **above** the items, `11-Nov-2025` dates, two-column header |
| `03_multipage_roundoff.pdf` | 30 line items over 2 pages (header repeated), discount, Round Off line, totals on page 2 |
| `04_scanned_rotated.pdf` | **Image-only** (no text layer): rasterized, rotated 1.5°, specks, blur, JPEG artifacts. Must go through OCR |
| `05_eu_vat_defective.pdf` | EU (German seller) invoice in EUR, `1.234,56` number format, DD.MM.YYYY, with **deliberate defects**: line items don't sum to the printed net total, malformed VAT ID, no invoice number, no due date |

## Writing an expected file

1. Copy `expected/_TEMPLATE.json` to `expected/<pdf-stem>.json` (e.g. `01_gst_classic.json`).
2. Open the PDF and fill in **what is printed**. Don't consult the generator script; reading the
   document like a human reviewer would is the whole point.
3. Rules:
   - `null` means "not printed on this document". Every key must be present.
   - Amounts and quantities are **JSON strings** in plain decimal form: `"123456.00"`, not `123456` or `"1,23,456.00"`.
     (A bare number is rejected, because floats lose precision.)
   - Dates are ISO: `"2025-11-05"`.
   - `currency` is the ISO code (`"INR"`, `"EUR"`), even if the document prints `Rs.` or `€`.
   - `tax_scheme`: `"gstin"`, `"eu_vat"`, `"other"`, `"unknown"`.
   - `adjustments[].kind`: `"discount"`, `"shipping"`, `"round_off"`, `"other"`. `amount` is **signed** by
     its effect on the total (a discount of 500 is `"-500.00"`).
   - `adjustments[].label` (optional): the label exactly as printed, e.g. `"Less: Trade Discount"`.
   - `taxable_value` (optional): only if a tax base is printed with its own label (e.g. "Taxable Value").
     Never derive it (e.g. subtotal minus discount). If it isn't printed, use `null` or leave the key out.
   - `tax_lines[].rate` is a percent: `"9"` for 9%.
   - `address`: the postal address only. The tax ID goes in `tax_id`, not here. When compared, runs of
     commas and whitespace count as one separator, so a line break may be written as a space or as ", ".
   - `subtotal`: the printed pre-tax total of the items. If the only such total is labelled "Taxable Value"
     (no discount), it is both `subtotal` and `taxable_value` (as in 02).
   - `expected_pages`: map from field path to page, e.g. `{"grand_total": 2, "line_items.29": 2}`.
     Every key is checked when the file loads. It must name a field in this format, with array indexes
     inside the list (0-based). Misspelled keys, out-of-range indexes, and metadata keys (`notes`, `fixture`) are rejected.
   - `expected_flags`: the `FlagCode` values validation should raise (see `app/models.py`), e.g.
     `["line_items_sum_mismatch", "tax_id_invalid_format", "required_field_missing"]`.
   - `notes`: anything a future reader should know ("Round Off is printed as R/O").
4. Check it loads: `.venv/bin/pytest tests/test_models.py -k expected`. Every file in `expected/` is
   validated against `ExpectedInvoice`.
