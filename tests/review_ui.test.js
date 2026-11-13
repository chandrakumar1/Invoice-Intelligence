/*
What:     Unit tests for the pure functions in static/review.js, run with Node's built-in test runner (no npm packages).
In:       A real result JSON produced by the pipeline (path in REVIEW_RESULT_JSON, written by tests/test_review_ui.py), plus literals.
Out:      node --test pass/fail; tests/test_review_ui.py runs this file and skips it when Node isn't installed.
Fails if: The result JSON shape changes and review.js doesn't follow, which is exactly what this catches.
*/
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ui = require(path.join(__dirname, "..", "static", "review.js"));

const result = JSON.parse(fs.readFileSync(process.env.REVIEW_RESULT_JSON, "utf8"));

test("every extracted value becomes a row, in document order", () => {
  const rows = ui.flattenRows(result.invoice);
  const paths = rows.map((r) => r.path);
  assert.equal(paths[0], "invoice_number");
  assert.equal(paths.at(-1), "grand_total");
  assert.ok(paths.includes("line_items.3") && paths.includes("tax_lines.0.rate") && paths.includes("seller.tax_id"));
  assert.deepEqual(rows.map((r) => r.order), [...rows.keys()]);
});

test("missing values have confidence 0 and keep their reason", () => {
  const row = ui.flattenRows(result.invoice).find((r) => r.path === "invoice_number");
  assert.equal(row.value, null);
  assert.equal(row.confidence, 0);
  assert.match(row.reason, /no invoice number/);
});

test("values are shown as printed-derived strings", () => {
  const rows = ui.flattenRows(result.invoice);
  const get = (p) => rows.find((r) => r.path === p).value;
  assert.equal(get("subtotal"), "1457.00");
  assert.equal(get("tax_lines.0.rate"), "19%");
  assert.equal(get("seller.tax_id"), "DE12345678 (EU VAT)");
  assert.equal(get("line_items.0"), "Desk Organizer Bamboo · code BO-1101 · 15 × 24.90 = 373.50");
});

test("flags colour the rows they name, not whole collections", () => {
  const flags = result.flags;
  assert.equal(ui.rowSeverity("invoice_number", flags), "error");
  assert.equal(ui.rowSeverity("subtotal", flags), "error");
  assert.equal(ui.rowSeverity("seller.tax_id", flags), "error");
  assert.equal(ui.rowSeverity("line_items.0", flags), null); // the sum flag names "line_items", the collection
  assert.equal(ui.rowSeverity("grand_total", flags), null);
});

test("flag path matching", () => {
  const f = (p, severity = "warning") => ({ field_paths: [p], severity });
  assert.equal(ui.flagsFor("line_items.3", [f("line_items.3.amount")]).length, 1); // a cell inside the row
  assert.equal(ui.flagsFor("tax_lines.0.amount", [f("tax_lines.0")]).length, 1); // the row's parent field
  assert.equal(ui.flagsFor("tax_lines.0.amount", [f("tax_lines")]).length, 0); // a whole collection
  assert.equal(ui.flagsFor("seller.name", [f("seller.tax_id")]).length, 0);
  assert.equal(ui.rowSeverity("x", [f("x", "warning"), f("x", "error")]), "error");
});

test("rows sort lowest confidence first, then errors, then document order", () => {
  const rows = [
    { path: "a", confidence: 1.0, order: 0 },
    { path: "b", confidence: 0.6, order: 1 },
    { path: "c", confidence: 1.0, order: 2 },
    { path: "d", confidence: 0, order: 3 },
  ];
  const flags = [{ field_paths: ["c"], severity: "error" }];
  assert.deepEqual(ui.sortRows(rows, flags).map((r) => r.path), ["d", "b", "c", "a"]);
});

test("sorting the real result puts missing values and errors on top", () => {
  const sorted = ui.sortRows(ui.flattenRows(result.invoice), result.flags);
  assert.equal(sorted[0].confidence, 0);
  const firstClean = sorted.findIndex((r) => r.confidence === 1 && r.severity === null);
  const lastErrorAtOne = sorted.findLastIndex((r) => r.confidence === 1 && r.severity === "error");
  assert.ok(lastErrorAtOne < firstClean);
});

test("highlight segments cover the quote exactly once", () => {
  const quote = "1 BO-1101 Desk Organizer Bamboo 15 24,90 € 373,50 €";
  const amount = quote.indexOf("373,50");
  // overlapping, unsorted, past-the-end and before-the-start spans
  const segs = ui.highlightSegments(quote, [[10, 31], [2, 9], [5, 12], [amount, 999], [-3, 0]]);
  assert.equal(segs.map((s) => s.text).join(""), quote);
  assert.deepEqual(segs.filter((s) => s.mark).map((s) => s.text), ["BO-1101 Desk Organizer Bamboo", "373,50 €"]);
  assert.deepEqual(ui.highlightSegments("abc", []), [{ text: "abc", mark: false }]);
});

test("summary counts", () => {
  const rows = ui.flattenRows(result.invoice);
  const s = ui.summarize(rows, result.flags);
  assert.equal(s.errors, 3);
  assert.equal(s.warnings, 0);
  assert.equal(s.total, rows.length);
  assert.ok(s.low >= 3); // invoice number, due date and taxable value are missing
});

test("element children are flattened to any depth, so nodes are never stringified", () => {
  // Shape renderPages passes: one array of [lineNumberSpan, text] pairs per page.
  const span = { tag: "span" };
  const kids = ui.flattenChildren([[[span, " first\n"], [span, " second\n"]]]);
  assert.deepEqual(kids, [span, " first\n", span, " second\n"]);
  assert.ok(kids.every((k) => !Array.isArray(k)));
  assert.deepEqual(ui.flattenChildren(["a", null, [undefined, [false, "b"]], 0, ""]), ["a", "b", 0, ""]);
});
