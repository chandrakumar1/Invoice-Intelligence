"""
What:     Checks each LLM citation against the canonical page text, slices the verbatim quote, and scores confidence.
In:       A value's text plus its page/line citation from the LLM, and the document's PageInfo list.
Out:      Located / RowLocation (page, lines, verbatim quote, value spans, match type, the value as printed), or None.
Fails if: Fuzzy matching accepts a near-miss. It needs 0.9 similarity, never runs on values under 4 characters or
          outside the cited lines, and the value is then read from the page, not from the LLM.

The rule throughout: the LLM only points. Every value handed downstream is a slice of the page text.
"""

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.models import MatchType, PageInfo, SourceRef, Span

MATCH_FACTOR = {
    MatchType.EXACT: 1.0,
    MatchType.NORMALIZED: 0.95,
    MatchType.RELOCATED: 0.75,
    MatchType.FUZZY: 0.6,
}
FUZZY_MIN_RATIO = 0.9
SHORT_VALUE = 4  # values shorter than this (normalized) must match at token boundaries and are never fuzzy-matched

_TRANSLATE = str.maketrans(
    {
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
        "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    }
)


@dataclass(frozen=True)
class Located:
    page: int
    line_start: int
    line_end: int
    quote: str  # the matched line(s), verbatim
    spans: tuple[Span, ...]  # one per segment of the value, as offsets into `quote`
    match_type: MatchType
    text: str  # the value exactly as printed: its segments joined with "\n"

    @property
    def span(self) -> Span:
        """Covers the whole value. For a multi-line value it can include text from a neighbouring column."""
        return (self.spans[0][0], self.spans[-1][1])


@dataclass(frozen=True)
class CellMatch:
    span: Span
    text: str
    match_type: MatchType


@dataclass(frozen=True)
class RowLocation:
    line_start: int
    line_end: int
    quote: str
    cells: dict[str, CellMatch]  # only the cells that were found

    @property
    def match_type(self) -> MatchType:
        return _worst(m.match_type for m in self.cells.values())


@dataclass(frozen=True)
class _Hit:
    line: int  # 1-indexed
    start: int  # offsets within that line
    end: int
    match_type: MatchType


# --- Public API ----------------------------------------------------------------------------


def locate(value_text: str, page: int | None, line_start: int | None, line_end: int | None, pages: list[PageInfo]) -> Located | None:
    """Find `value_text` on the cited lines; failing that (single-line values only), anywhere in the document."""
    segments = _segments(value_text)
    if not segments:
        return None
    by_number = {p.page: p for p in pages}
    cited = by_number.get(page) if page is not None else None
    if cited and _valid_range(cited, line_start, line_end):
        if hits := _match_segments(segments, cited, line_start, line_end, claimed={}):
            return _located(cited, hits)
    if len(segments) == 1:
        if hit_page := _relocate(segments[0], pages, cited, line_start or 1):
            found_page, hit = hit_page
            return _located(found_page, [hit])
    return None


def locate_row(cells: dict[str, str], page: PageInfo, line_start: int, line_end: int) -> RowLocation | None:
    """Find each cell of one table row within the row's cited lines. None if the range is invalid or no cell is found."""
    if not _valid_range(page, line_start, line_end):
        return None
    claimed: dict[int, list[Span]] = {}
    hits: dict[str, list[_Hit]] = {}
    for cell, text in sorted(cells.items(), key=lambda kv: -len(kv[1])):  # long values claim their spot first
        segments = _segments(text)
        if segments and (found := _match_segments(segments, page, line_start, line_end, claimed)):
            hits[cell] = found
    if not hits:
        return None
    lo, hi, quote, offsets = _quote(page, [h for group in hits.values() for h in group])
    matches = {}
    for cell, group in hits.items():
        spans = [(offsets[h.line] + h.start, offsets[h.line] + h.end) for h in group]
        text = "\n".join(page.lines[h.line - 1][h.start : h.end] for h in group)
        matches[cell] = CellMatch((spans[0][0], spans[-1][1]), text, _worst(h.match_type for h in group))
    return RowLocation(lo, hi, quote, matches)


def confidence(match_type: MatchType, page: PageInfo, line_start: int, line_end: int, certainty: float = 1.0) -> float:
    """match factor x OCR factor x parse certainty. A heuristic score in [0, 1], not a probability."""
    return round(MATCH_FACTOR[match_type] * ocr_factor(page, line_start, line_end) * certainty, 3)


def ocr_factor(page: PageInfo, line_start: int, line_end: int) -> float:
    """1.0 for a text layer; for OCR, the mean Tesseract confidence of the quoted lines, scaled to [0, 1]."""
    if page.text_source == "text_layer":
        return 1.0
    confs = (page.ocr_line_conf or [])[line_start - 1 : line_end]
    return sum(confs) / len(confs) / 100 if confs else 0.0


def source_ref(loc: Located, page: PageInfo, certainty: float = 1.0) -> SourceRef:
    return SourceRef(
        page=loc.page,
        line_start=loc.line_start,
        line_end=loc.line_end,
        quote=loc.quote,
        value_span=loc.span,
        match_type=loc.match_type,
        text_source=page.text_source,
        confidence=confidence(loc.match_type, page, loc.line_start, loc.line_end, certainty),
    )


# --- Matching ------------------------------------------------------------------------------


def _segments(value_text: str) -> list[str]:
    return [s.strip() for s in value_text.split("\n") if s.strip()]


def _valid_range(page: PageInfo, line_start: int | None, line_end: int | None) -> bool:
    return line_start is not None and line_end is not None and 1 <= line_start <= line_end <= len(page.lines)


def _match_segments(segments: list[str], page: PageInfo, line_start: int, line_end: int, claimed: dict[int, list[Span]]) -> list[_Hit] | None:
    """Each segment on its own line, in order, within the range. Two-column layouts can put other text on the same lines."""
    hits: list[_Hit] = []
    next_line = line_start
    for segment in segments:
        for n in range(next_line, line_end + 1):
            if found := _find(segment, page.lines[n - 1], claimed.get(n, [])):
                hits.append(_Hit(n, *found))
                next_line = n + 1
                break
        else:
            return None
    for h in hits:
        claimed.setdefault(h.line, []).append((h.start, h.end))
    return hits


def _relocate(segment: str, pages: list[PageInfo], cited: PageInfo | None, near: int) -> tuple[PageInfo, _Hit] | None:
    """Search the whole document: the cited page first (nearest lines first), then the others in order. No fuzzy."""
    order = ([cited] if cited else []) + [p for p in pages if p is not cited]
    for page in order:
        numbers = range(1, len(page.lines) + 1)
        if page is cited:
            numbers = sorted(numbers, key=lambda n: (abs(n - near), n))
        for n in numbers:
            found = _find(segment, page.lines[n - 1], [], allow_fuzzy=False)
            if found:
                start, end, _ = found
                return page, _Hit(n, start, end, MatchType.RELOCATED)
    return None


def _find(needle: str, line: str, claimed: list[Span], allow_fuzzy: bool = True) -> tuple[int, int, MatchType] | None:
    norm_needle, _ = _normalize(needle)
    if not norm_needle:
        return None
    short = len(norm_needle) < SHORT_VALUE

    exact = _occurrences(line, needle)
    if pick := _pick(exact, line, claimed, short):
        return (*pick, MatchType.EXACT)

    norm_line, back = _normalize(line)
    normalized = [(back[k], back[k + len(norm_needle) - 1] + 1) for k in _occurrence_starts(norm_line, norm_needle)]
    if pick := _pick(normalized, line, claimed, short):
        return (*pick, MatchType.NORMALIZED)

    if allow_fuzzy and not short and (window := _fuzzy(norm_needle, norm_line, back, claimed, line)):
        return (*window, MatchType.FUZZY)
    return None


def _pick(candidates: list[Span], line: str, claimed: list[Span], short: bool) -> Span | None:
    free = [c for c in candidates if not any(_overlaps(c, taken) for taken in claimed)]
    bounded = [c for c in free if _at_boundaries(line, *c)]
    if bounded:
        return bounded[0]
    return None if short or not free else free[0]


def _fuzzy(norm_needle: str, norm_line: str, back: list[int], claimed: list[Span], line: str) -> Span | None:
    """Best window of similar length, starting and ending on token boundaries, so a value is never cut mid-token
    (otherwise dropping one mismatched character can outscore the whole printed value)."""
    best_ratio, best = 0.0, None
    size = len(norm_needle)
    for width in range(max(1, size - 2), size + 3):
        for i in range(0, len(norm_line) - width + 1):
            span = (back[i], back[i + width - 1] + 1)
            if not _at_boundaries(line, *span) or any(_overlaps(span, taken) for taken in claimed):
                continue
            ratio = SequenceMatcher(None, norm_needle, norm_line[i : i + width], autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, span
    return best if best_ratio >= FUZZY_MIN_RATIO else None


def _normalize(s: str) -> tuple[str, list[int]]:
    """NFKC, unified dashes/quotes, casefolded, whitespace collapsed; plus a map back to offsets in `s`."""
    out: list[str] = []
    back: list[int] = []
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKC", ch).translate(_TRANSLATE).casefold():
            if c.isspace():
                if not out or out[-1] == " ":
                    continue
                c = " "
            out.append(c)
            back.append(i)
    while out and out[-1] == " ":
        out.pop()
        back.pop()
    return "".join(out), back


def _occurrences(haystack: str, needle: str) -> list[Span]:
    return [(k, k + len(needle)) for k in _occurrence_starts(haystack, needle)]


def _occurrence_starts(haystack: str, needle: str) -> list[int]:
    starts, k = [], haystack.find(needle)
    while k != -1:
        starts.append(k)
        k = haystack.find(needle, k + 1)
    return starts


def _at_boundaries(line: str, start: int, end: int) -> bool:
    return (start == 0 or not line[start - 1].isalnum()) and (end == len(line) or not line[end].isalnum())


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _worst(types) -> MatchType:
    return min(types, key=lambda t: MATCH_FACTOR[t])


def _quote(page: PageInfo, hits: list[_Hit]) -> tuple[int, int, str, dict[int, int]]:
    lo, hi = min(h.line for h in hits), max(h.line for h in hits)
    offsets, pos = {}, 0
    for n in range(lo, hi + 1):
        offsets[n] = pos
        pos += len(page.lines[n - 1]) + 1
    return lo, hi, "\n".join(page.lines[lo - 1 : hi]), offsets


def _located(page: PageInfo, hits: list[_Hit]) -> Located:
    lo, hi, quote, offsets = _quote(page, hits)
    spans = tuple((offsets[h.line] + h.start, offsets[h.line] + h.end) for h in hits)
    text = "\n".join(page.lines[h.line - 1][h.start : h.end] for h in hits)
    return Located(page.page, lo, hi, quote, spans, _worst(h.match_type for h in hits), text)
