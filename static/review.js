/*
What:     Review UI logic: upload a PDF, poll for the result, and show every extracted value beside its verbatim source quote.
In:       ?id=<job id> in the page URL (or a file chosen in the upload form); JSON from GET /result/{id}.
Out:      DOM. Rows sorted lowest-confidence first; rows named by an error flag in red, by a warning in amber.
Fails if: The result JSON shape changes without this file following. tests/review_ui.test.js covers the pure functions.

All document text goes through textContent, never innerHTML: invoice text is untrusted input.
*/
"use strict";

const LOW_CONFIDENCE = 0.8;
const POLL_MS = 1500;
const POLL_LIMIT_MS = 15 * 60 * 1000;

// ---------------------------------------------------------------------------------------------
// Pure logic (no DOM): exported for tests/review_ui.test.js
// ---------------------------------------------------------------------------------------------

const SCHEMES = { gstin: "GSTIN", eu_vat: "EU VAT", other: "other", unknown: "unknown scheme" };

/** One review row per extracted value, in document order. Missing values get confidence 0. */
function flattenRows(inv) {
  const rows = [];
  const add = (path, label, field, show = String) => {
    const present = field.value !== null && field.value !== undefined;
    rows.push({
      path,
      label,
      value: present ? show(field.value) : null,
      reason: field.missing_reason || null,
      source: field.source || null,
      spans: field.source && field.source.value_span ? [field.source.value_span] : [],
      confidence: present ? field.source.confidence : 0,
      notes: [],
      order: rows.length,
    });
  };

  add("invoice_number", "Invoice number", inv.invoice_number);
  add("invoice_date", "Invoice date", inv.invoice_date);
  add("due_date", "Due date", inv.due_date);
  add("currency", "Currency", inv.currency);
  for (const side of ["seller", "buyer"]) {
    const Side = side[0].toUpperCase() + side.slice(1);
    add(`${side}.name`, `${Side} name`, inv[side].name);
    add(`${side}.address`, `${Side} address`, inv[side].address);
    add(`${side}.tax_id`, `${Side} tax ID`, inv[side].tax_id, (t) => `${t.raw} (${SCHEMES[t.scheme] || t.scheme})`);
  }
  inv.line_items.forEach((li, i) => {
    rows.push({
      path: `line_items.${i}`,
      label: `Line ${i + 1}`,
      value: lineSummary(li),
      reason: null,
      source: li.source,
      spans: Object.values(li.cell_spans),
      confidence: li.source.confidence,
      notes: Object.entries(li.missing_reasons)
        .filter(([, why]) => why !== "not printed in this row")
        .map(([cell, why]) => `${cell}: ${why}`),
      order: rows.length,
    });
  });
  add("subtotal", "Subtotal", inv.subtotal);
  add("taxable_value", "Taxable value", inv.taxable_value);
  inv.tax_lines.forEach((t, i) => {
    add(`tax_lines.${i}.label`, `Tax line ${i + 1}: label`, t.label);
    add(`tax_lines.${i}.rate`, `Tax line ${i + 1}: rate`, t.rate, (r) => `${r}%`);
    add(`tax_lines.${i}.amount`, `Tax line ${i + 1}: amount`, t.amount);
  });
  add("tax_total", "Tax total", inv.tax_total);
  inv.adjustments.forEach((a, i) => {
    const kind = a.kind.replace("_", " ");
    add(`adjustments.${i}.label`, `Adjustment ${i + 1} (${kind}): label`, a.label);
    add(`adjustments.${i}.amount`, `Adjustment ${i + 1} (${kind}): amount`, a.amount);
  });
  add("grand_total", "Grand total", inv.grand_total);
  return rows;
}

function lineSummary(li) {
  const parts = [];
  if (li.description !== null) parts.push(li.description);
  if (li.hsn_sac !== null) parts.push(`code ${li.hsn_sac}`);
  const qty = li.quantity !== null ? li.quantity : "?";
  const price = li.unit_price !== null ? li.unit_price : "?";
  const product = li.quantity !== null || li.unit_price !== null ? `${qty} × ${price}` : null;
  const amount = li.amount !== null ? li.amount : null;
  if (product && amount) parts.push(`${product} = ${amount}`);
  else if (product || amount) parts.push(product || amount);
  return parts.join(" · ");
}

/**
 * Flags that name this row. A flag path matches the row itself, a field inside it, or the row's parent
 * field (e.g. "seller.tax_id" for that row). Collection-level paths like "line_items" don't colour every
 * row; those flags are listed in the flags panel instead.
 */
function flagsFor(path, flags) {
  return flags.filter((f) =>
    f.field_paths.some((p) => p === path || p.startsWith(path + ".") || (p.includes(".") && path.startsWith(p + ".")))
  );
}

function rowSeverity(path, flags) {
  const matched = flagsFor(path, flags);
  if (matched.some((f) => f.severity === "error")) return "error";
  if (matched.some((f) => f.severity === "warning")) return "warning";
  return null;
}

const SEVERITY_RANK = { error: 0, warning: 1, null: 2 };

/** Lowest confidence first; among equals, errors before warnings before clean rows, then document order. */
function sortRows(rows, flags) {
  return rows
    .map((r) => ({ ...r, severity: rowSeverity(r.path, flags) }))
    .sort((a, b) => a.confidence - b.confidence || SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || a.order - b.order);
}

/** Split a quote into plain and highlighted pieces. Spans may overlap, be unsorted, or exceed the quote. */
function highlightSegments(quote, spans) {
  const clean = spans
    .map(([s, e]) => [Math.max(0, s), Math.min(quote.length, e)])
    .filter(([s, e]) => s < e)
    .sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of clean) {
    const last = merged[merged.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  const out = [];
  let pos = 0;
  for (const [s, e] of merged) {
    if (s > pos) out.push({ text: quote.slice(pos, s), mark: false });
    out.push({ text: quote.slice(s, e), mark: true });
    pos = e;
  }
  if (pos < quote.length) out.push({ text: quote.slice(pos), mark: false });
  return out;
}

function summarize(rows, flags) {
  return {
    errors: flags.filter((f) => f.severity === "error").length,
    warnings: flags.filter((f) => f.severity === "warning").length,
    low: rows.filter((r) => r.confidence < LOW_CONFIDENCE).length,
    total: rows.length,
  };
}

// ---------------------------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------------------------

/** Build an element. Children are nodes or strings; strings become text nodes (never parsed as HTML). */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function $(id) {
  return document.getElementById(id);
}

function show(id, visible) {
  $(id).hidden = !visible;
}

function setStatus(text, kind = "info") {
  const box = $("status");
  box.replaceChildren(text);
  box.className = `status ${kind}`;
  box.hidden = !text;
}

function renderHeader(result) {
  const created = new Date(result.created_at).toLocaleString();
  $("doc-title").textContent = result.filename;
  $("doc-meta").replaceChildren(
    el("span", { class: `pill ${result.status}` }, result.status),
    ` uploaded ${created}`,
    result.model ? ` · ${result.provider ? `${result.provider} ` : ""}model ${result.model}` : null,
    result.prompt_version ? ` · prompt ${result.prompt_version}` : null,
    ` · pipeline ${result.pipeline_version}`
  );
  show("doc-header", true);
}

function renderSummary(s) {
  $("summary").replaceChildren(
    el("div", { class: `stat ${s.errors ? "error" : ""}` }, el("strong", {}, String(s.errors)), s.errors === 1 ? " error" : " errors"),
    el("div", { class: `stat ${s.warnings ? "warning" : ""}` }, el("strong", {}, String(s.warnings)), s.warnings === 1 ? " warning" : " warnings"),
    el("div", { class: `stat ${s.low ? "warning" : ""}` }, el("strong", {}, String(s.low)), ` of ${s.total} values below ${LOW_CONFIDENCE} confidence`)
  );
}

function renderFlags(flags) {
  const list = $("flags");
  if (!flags.length) {
    list.replaceChildren(el("li", { class: "ok" }, "No validation problems."));
    return;
  }
  const ordered = [...flags].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);
  list.replaceChildren(
    ...ordered.map((f) =>
      el(
        "li",
        { class: f.severity },
        el("span", { class: `badge ${f.severity}` }, f.severity === "error" ? "Error" : "Warning"),
        " ",
        f.message,
        el("span", { class: "paths" }, ` (${f.field_paths.join(", ")})`)
      )
    )
  );
}

function quoteCell(row) {
  if (!row.source) return el("td", { class: "quote missing" }, "no source");
  const src = row.source;
  const lines = src.line_start === src.line_end ? `line ${src.line_start}` : `lines ${src.line_start}–${src.line_end}`;
  const pre = el(
    "pre",
    {},
    highlightSegments(src.quote, row.spans).map((seg) => (seg.mark ? el("mark", {}, seg.text) : seg.text))
  );
  return el(
    "td",
    { class: "quote" },
    pre,
    el(
      "div",
      { class: "where" },
      `page ${src.page}, ${lines}`,
      src.match_type !== "exact" ? el("span", { class: "badge muted" }, src.match_type) : null,
      src.text_source === "ocr" ? el("span", { class: "badge muted" }, "OCR") : null
    )
  );
}

function confidenceCell(row) {
  const pct = Math.round(row.confidence * 100);
  const low = row.confidence < LOW_CONFIDENCE;
  return el(
    "td",
    { class: `conf ${low ? "low" : ""}` },
    el("span", {}, row.confidence.toFixed(2)),
    el("div", { class: "bar", role: "presentation" }, el("div", { style: `width:${pct}%` }))
  );
}

function renderRows(rows) {
  $("rows").replaceChildren(
    ...rows.map((row) =>
      el(
        "tr",
        { class: row.severity || "" },
        el(
          "th",
          { scope: "row" },
          row.label,
          row.severity ? el("span", { class: `badge ${row.severity}` }, row.severity === "error" ? "Error" : "Warning") : null
        ),
        el(
          "td",
          { class: "value" },
          row.value !== null ? el("span", {}, row.value) : el("span", { class: "missing" }, "—"),
          row.reason ? el("div", { class: "reason" }, row.reason) : null,
          row.notes.map((n) => el("div", { class: "reason" }, n))
        ),
        quoteCell(row),
        confidenceCell(row)
      )
    )
  );
}

function renderPages(pages) {
  $("pages").replaceChildren(
    ...pages.map((p) =>
      el(
        "details",
        {},
        el(
          "summary",
          {},
          `Page ${p.page} · ${p.text_source === "ocr" ? `OCR (mean confidence ${Math.round(p.ocr_mean_conf)})` : "text layer"} · ${p.text ? p.text.split("\n").length : 0} lines`
        ),
        p.ocr_reasons.length ? el("p", { class: "reason" }, `Sent to OCR because: ${p.ocr_reasons.join("; ")}`) : null,
        el(
          "pre",
          { class: "page-text" },
          (p.text ? p.text.split("\n") : []).map((line, i) => [el("span", { class: "ln" }, String(i + 1).padStart(3, " ")), ` ${line}\n`])
        )
      )
    )
  );
  show("pages-section", pages.length > 0);
}

function renderResult(result) {
  renderHeader(result);
  if (result.status === "failed") {
    setStatus(`Extraction failed: ${result.error}`, "error");
  } else {
    setStatus("");
  }
  if (result.invoice) {
    const rows = sortRows(flattenRows(result.invoice), result.flags);
    renderSummary(summarize(rows, result.flags));
    renderFlags(result.flags);
    renderRows(rows);
    show("review", true);
  }
  renderPages(result.pages || []);
}

// ---------------------------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------------------------

async function poll(id, startedAt = Date.now()) {
  let result;
  try {
    const response = await fetch(`/result/${encodeURIComponent(id)}`);
    if (response.status === 404) return setStatus("No result with this id.", "error");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    result = await response.json();
  } catch (err) {
    return setStatus(`Couldn't load the result (${err.message}). Reload the page to try again.`, "error");
  }
  if (result.status === "pending") {
    renderHeader(result);
    if (Date.now() - startedAt > POLL_LIMIT_MS) return setStatus("Still processing after 15 minutes. Reload the page to check again.", "warning");
    setStatus(`Extracting… ${Math.round((Date.now() - startedAt) / 1000)} s`);
    return setTimeout(() => poll(id, startedAt), POLL_MS);
  }
  renderResult(result);
}

async function upload(event) {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  setStatus("Uploading…");
  $("upload-button").disabled = true;
  try {
    const response = await fetch("/extract", { method: "POST", body });
    const data = await response.json();
    if (response.status !== 202) throw new Error(data.detail || `HTTP ${response.status}`);
    history.replaceState(null, "", `?id=${data.id}`);
    show("upload", false);
    poll(data.id);
  } catch (err) {
    setStatus(`Upload failed: ${err.message}`, "error");
  } finally {
    $("upload-button").disabled = false;
  }
}

function boot() {
  const id = new URLSearchParams(location.search).get("id");
  $("upload-form").addEventListener("submit", upload);
  if (id) {
    show("upload", false);
    poll(id);
  }
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", boot);
if (typeof module !== "undefined") {
  module.exports = { LOW_CONFIDENCE, flattenRows, lineSummary, flagsFor, rowSeverity, sortRows, highlightSegments, summarize };
}
