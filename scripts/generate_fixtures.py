"""
What:     Generates the five synthetic invoice PDFs in tests/fixtures/pdfs/, byte-identical on every run.
In:       Optional `--out DIR` (default: tests/fixtures/pdfs). No network, no randomness beyond fixed seeds.
Out:      01_*.pdf ... 05_*.pdf. Deliberately no expected JSON: ground truth is hand-written from the rendered PDFs.
Fails if: A reportlab/Pillow/pdfium upgrade changes the output bytes. tests/test_fixtures.py then fails; regenerate and re-check the PDFs by eye.

Layouts are varied on purpose (see tests/fixtures/README.md). Defects in 05 are deliberate.
"""

import argparse
import io
import random
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable

import pdfplumber
from PIL import Image, ImageDraw, ImageFilter
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

rl_config.invariant = 1  # fixed timestamps and document IDs, so the output bytes are reproducible

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "pdfs"
CENT = Decimal("0.01")
FIXED_DATE = datetime(2025, 11, 1, 12, 0, 0)

STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=STYLES["Normal"], fontSize=9, leading=11)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9, textColor=colors.HexColor("#444444"))
H_TITLE = ParagraphStyle("title", parent=STYLES["Title"], fontSize=16, spaceAfter=4)
BOLD = ParagraphStyle("bold", parent=BODY, fontName="Helvetica-Bold")


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin(state: str, pan: str, entity: str = "1") -> str:
    """Build a GSTIN with a valid Luhn mod-36 check digit (verified against published GSTINs)."""
    body = f"{state}{pan}{entity}Z"
    factor, total = 2, 0
    for ch in reversed(body):
        addend = factor * GSTIN_CHARS.index(ch)
        factor = 1 if factor == 2 else 2
        total += addend // 36 + addend % 36
    return body + GSTIN_CHARS[(36 - total % 36) % 36]


@dataclass(frozen=True)
class Item:
    description: str
    code: str
    qty: Decimal
    rate: Decimal

    @property
    def amount(self) -> Decimal:
        return (self.qty * self.rate).quantize(CENT, ROUND_HALF_UP)


@dataclass(frozen=True)
class PartyInfo:
    name: str
    address: list[str]
    tax_id_label: str
    tax_id: str


def item(description: str, code: str, qty: int | str, rate: str) -> Item:
    return Item(description, code, Decimal(qty), Decimal(rate))


def pct(base: Decimal, rate: str) -> Decimal:
    return (base * Decimal(rate) / 100).quantize(CENT, ROUND_HALF_UP)


def fmt_inr(d: Decimal) -> str:
    """Indian digit grouping: 1,23,456.00"""
    sign = "-" if d < 0 else ""
    whole, frac = f"{abs(d):.2f}".split(".")
    if len(whole) > 3:
        head, tail, groups = whole[:-3], whole[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join([head, *groups, tail]) if head else ",".join([*groups, tail])
    return f"{sign}{whole}.{frac}"


def fmt_western(d: Decimal) -> str:
    return f"{d:,.2f}"


def fmt_eu(d: Decimal) -> str:
    return f"{d:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_qty(q: Decimal) -> str:
    return f"{q:f}"


# --------------------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------------------


def make_doc(target: str | Path | io.BytesIO, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(target) if isinstance(target, Path) else target,  # reportlab rejects Path objects
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Invoice Intelligence fixture generator",
        subject="Synthetic test invoice",
        creator="scripts/generate_fixtures.py",
    )


def page_footer(text: str) -> Callable:
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 10 * mm, text)
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def party_block(heading: str, p: PartyInfo) -> list:
    lines = "<br/>".join(p.address)
    return [
        Paragraph(f"<b>{heading}</b>", SMALL),
        Paragraph(f"<b>{p.name}</b>", BODY),
        Paragraph(lines, BODY),
        Paragraph(f"{p.tax_id_label}: {p.tax_id}", BODY),
    ]


def grid_style(header_bg: str = "#E8E8E8", numeric_from: int = 3) -> TableStyle:
    return TableStyle(
        [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
            ("ALIGN", (numeric_from, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def totals_table(rows: list[tuple[str, str]], bold_last: bool = True) -> Table:
    t = Table(rows, colWidths=[150, 100], hAlign="RIGHT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if bold_last:
        style.append(("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 10))
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------------------
# Classic Indian GST layout (used by 01 and, rasterized, by 04)
# --------------------------------------------------------------------------------------


def build_classic_gst(
    target: str | Path | io.BytesIO,
    *,
    seller: PartyInfo,
    buyer: PartyInfo,
    meta: list[tuple[str, str]],
    items: list[Item],
    taxes: list[tuple[str, str]],
    footer: str,
) -> None:
    subtotal = sum((i.amount for i in items), Decimal("0"))
    tax_amounts = [(label, pct(subtotal, rate)) for label, rate in taxes]
    grand = subtotal + sum(a for _, a in tax_amounts)

    story: list = [Paragraph("TAX INVOICE", H_TITLE)]
    story += party_block("Supplier", seller)
    story.append(Spacer(1, 6))
    meta_t = Table([[k, v] for k, v in meta], colWidths=[90, 160], hAlign="LEFT")
    meta_t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 9), ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9)]))
    story += [meta_t, Spacer(1, 8)]
    story += party_block("Bill To", buyer)
    story.append(Spacer(1, 10))

    rows = [["#", "Description", "HSN/SAC", "Qty", "Rate (Rs.)", "Amount (Rs.)"]]
    for n, i in enumerate(items, 1):
        rows.append([str(n), i.description, i.code, fmt_qty(i.qty), fmt_inr(i.rate), fmt_inr(i.amount)])
    t = Table(rows, colWidths=[20, 215, 60, 40, 80, 95], repeatRows=1)
    t.setStyle(grid_style())
    story += [t, Spacer(1, 8)]

    trows = [("Sub Total", fmt_inr(subtotal))]
    trows += [(label, fmt_inr(a)) for label, a in tax_amounts]
    trows.append(("Total (Rs.)", fmt_inr(grand)))
    story += [totals_table(trows), Spacer(1, 16)]
    story.append(Paragraph("Bank: State Bank of India, A/c No. 30219847561, IFSC SBIN0001234", SMALL))
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"For {seller.name}<br/><br/>Authorised Signatory", BODY))

    doc = make_doc(target, "Tax Invoice")
    doc.build(story, onFirstPage=page_footer(footer), onLaterPages=page_footer(footer))


def make_01(out: Path) -> None:
    seller = PartyInfo(
        "Sharma Office Supplies Pvt. Ltd.",
        ["No. 42, 3rd Cross, Indiranagar", "Bengaluru, Karnataka 560038"],
        "GSTIN",
        gstin("29", "AAGCS4821K"),
    )
    buyer = PartyInfo(
        "Kaveri Software Solutions LLP",
        ["Plot 17, Electronic City Phase 1", "Bengaluru, Karnataka 560100"],
        "GSTIN",
        gstin("29", "AAKFK7730M"),
    )
    items = [
        item("A4 Copier Paper 75 GSM (500 sheets)", "4802", 40, "285.00"),
        item("Ball Pen Blue (Box of 50)", "9608", 12, "410.00"),
        item("Stapler Heavy Duty HD-45", "8472", 5, "1250.00"),
        item("Whiteboard Marker Assorted (Pack of 10)", "9608", 20, "365.00"),
        item("Ergonomic Office Chair Model EC-220", "9401", 3, "12499.00"),
    ]
    build_classic_gst(
        out / "01_gst_classic.pdf",
        seller=seller,
        buyer=buyer,
        meta=[
            ("Invoice No.", "SOS/2025-26/0147"),
            ("Invoice Date", "05/11/2025"),
            ("Due Date", "20/11/2025"),
            ("Place of Supply", "Karnataka (29)"),
        ],
        items=items,
        taxes=[("CGST @ 9%", "9"), ("SGST @ 9%", "9")],
        footer="Sharma Office Supplies Pvt. Ltd. | This is a computer generated invoice",
    )


# --------------------------------------------------------------------------------------
# 02: inter-state IGST, two-column header, totals box ABOVE the items
# --------------------------------------------------------------------------------------


def make_02(out: Path) -> None:
    seller = PartyInfo(
        "Deccan Packaging Industries",
        ["Gat No. 118, Chakan MIDC", "Pune, Maharashtra 410501"],
        "GSTIN",
        gstin("27", "AAEFD5512Q"),
    )
    buyer = PartyInfo(
        "Nilgiri Foods Pvt. Ltd.",
        ["14 Industrial Suburb, Yeshwanthpur", "Bengaluru, Karnataka 560022"],
        "GSTIN",
        gstin("29", "AACCN6274H"),
    )
    items = [
        item("Corrugated Box 5-Ply 18x12x12 in", "4819", 1500, "38.50"),
        item("Corrugated Box 3-Ply 12x10x8 in", "4819", 2500, "17.25"),
        item("BOPP Tape 48mm x 65m Brown", "3919", 240, "42.00"),
        item("Stretch Wrap Film 23 Micron 500mm", "3920", 60, "690.00"),
    ]
    subtotal = sum((i.amount for i in items), Decimal("0"))
    igst = pct(subtotal, "18")
    total = subtotal + igst

    left = party_block("From", seller)
    right = [
        Paragraph("<b>INVOICE</b>", ParagraphStyle("inv", parent=BODY, fontSize=18, leading=22)),
        Paragraph("Invoice #: DPI-INV-2025-0932", BODY),
        Paragraph("Issued: 11-Nov-2025", BODY),
        Paragraph("Payment due: 11-Dec-2025", BODY),
        Paragraph("Place of supply: Karnataka (29)", BODY),
    ]
    header = Table([[left, right]], colWidths=[290, 220])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    summary = Table(
        [
            ["Amount Due", "Taxable Value", "IGST @ 18%", "Invoice Total"],
            [f"INR {fmt_western(total)}", f"INR {fmt_western(subtotal)}", f"INR {fmt_western(igst)}", f"INR {fmt_western(total)}"],
        ],
        colWidths=[127, 128, 127, 128],
    )
    summary.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#1F4E79")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 11),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    rows = [["Item", "HSN", "Qty", "Unit Price", "Line Total"]]
    rows += [[i.description, i.code, fmt_qty(i.qty), fmt_western(i.rate), fmt_western(i.amount)] for i in items]
    t = Table(rows, colWidths=[230, 50, 60, 80, 90])
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#1F4E79")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story = [header, Spacer(1, 12), summary, Spacer(1, 14)]
    story += party_block("Bill To / Ship To", buyer)
    story += [Spacer(1, 12), t, Spacer(1, 18)]
    story.append(Paragraph("Terms: Payment within 30 days by NEFT/RTGS. Goods once sold will not be taken back.", SMALL))

    doc = make_doc(out / "02_igst_totals_top.pdf", "Invoice DPI-INV-2025-0932")
    footer = page_footer("Deccan Packaging Industries, Pune")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------------------
# 03: 30 items across two pages, discount, round off
# --------------------------------------------------------------------------------------

ELECTRICAL_ITEMS = [
    ("Modular Switch 6A 1-Way", "8536"), ("Modular Socket 6A 3-Pin", "8536"),
    ("MCB 16A Single Pole C-Curve", "8536"), ("MCB 32A Double Pole C-Curve", "8536"),
    ("RCCB 40A 4-Pole 30mA", "8536"), ("Distribution Board 8-Way SPN", "8537"),
    ("FR PVC Wire 1.5 sq mm (90 m)", "8544"), ("FR PVC Wire 2.5 sq mm (90 m)", "8544"),
    ("FR PVC Wire 4.0 sq mm (90 m)", "8544"), ("Armoured Cable 4C x 16 sq mm (per m)", "8544"),
    ("LED Bulb 9W Cool Daylight", "8539"), ("LED Batten 20W 4 ft", "9405"),
    ("LED Panel 15W Round Recessed", "9405"), ("LED Flood Light 50W IP65", "9405"),
    ("Ceiling Fan 1200mm White", "8414"), ("Exhaust Fan 250mm", "8414"),
    ("PVC Conduit Pipe 25mm (3 m)", "3917"), ("PVC Conduit Bend 25mm", "3917"),
    ("Junction Box 4-Way 25mm", "3925"), ("Modular Plate 8M White", "3925"),
    ("Metal Box 8M Concealed", "7326"), ("Earthing Electrode Copper Bonded 3 m", "8536"),
    ("Cable Tie 200mm (Pack of 100)", "3926"), ("Insulation Tape 18mm Black", "3919"),
    ("Screw Self-Tapping 25mm (Box of 200)", "7318"), ("Rawl Plug 6mm (Pack of 100)", "3926"),
    ("Changeover Switch 63A 4-Pole", "8536"), ("Doorbell Ding-Dong 230V", "8531"),
    ("Industrial Socket 32A 5-Pin IP44", "8536"), ("Copper Lug 16 sq mm (Pack of 50)", "8536"),
]


def make_03(out: Path) -> None:
    rng = random.Random(2025)
    items = [
        item(desc, code, rng.choice([2, 4, 5, 6, 10, 12, 20, 25, 50, 100]), str(Decimal(rng.randint(1500, 250000)) / 100))
        for desc, code in ELECTRICAL_ITEMS
    ]
    seller = PartyInfo(
        "Ganga Electricals & Hardware",
        ["21, Ganesh Peth", "Nagpur, Maharashtra 440018"],
        "GSTIN",
        gstin("27", "AAJFG3398C"),
    )
    buyer = PartyInfo(
        "Vidarbha Construction Co.",
        ["Near Zero Mile, Civil Lines", "Nagpur, Maharashtra 440001"],
        "GSTIN",
        gstin("27", "AAAFV8817D"),
    )
    subtotal = sum((i.amount for i in items), Decimal("0"))
    discount = Decimal("1500.00")
    taxable = subtotal - discount
    cgst, sgst = pct(taxable, "9"), pct(taxable, "9")
    exact = taxable + cgst + sgst
    grand = exact.quantize(Decimal("1"), ROUND_HALF_UP)
    round_off = grand - exact
    assert round_off != 0, "fixture 03 must exercise a non-zero Round Off"

    story: list = [Paragraph("Ganga Electricals &amp; Hardware", H_TITLE)]
    story.append(Paragraph("21, Ganesh Peth, Nagpur, Maharashtra 440018 | GSTIN: " + seller.tax_id, BODY))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>TAX INVOICE</b>", ParagraphStyle("c", parent=BODY, alignment=1, fontSize=12, leading=14)))
    story.append(Spacer(1, 6))
    info = Table(
        [
            [party_block("Buyer", buyer), [
                Paragraph("Invoice No: GEH/1124/2025", BODY),
                Paragraph("Date: 07/11/2025", BODY),
                Paragraph("Due Date: 07/12/2025", BODY),
                Paragraph("Order Ref: VCC/PO/7781", BODY),
            ]]
        ],
        colWidths=[300, 210],
    )
    info.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [info, Spacer(1, 10)]

    rows = [["Sr.", "Description of Goods", "HSN", "Qty", "Rate", "Amount"]]
    rows += [[str(n), i.description, i.code, fmt_qty(i.qty), fmt_inr(i.rate), fmt_inr(i.amount)] for n, i in enumerate(items, 1)]
    t = Table(rows, colWidths=[25, 225, 45, 40, 80, 95], repeatRows=1)
    style = grid_style("#D9E2F3")
    style.add("TOPPADDING", (0, 0), (-1, -1), 6)
    style.add("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    t.setStyle(style)
    story += [t, Spacer(1, 8)]

    story.append(
        totals_table(
            [
                ("Sub Total", fmt_inr(subtotal)),
                ("Less: Trade Discount", fmt_inr(-discount)),
                ("Taxable Value", fmt_inr(taxable)),
                ("CGST @ 9%", fmt_inr(cgst)),
                ("SGST @ 9%", fmt_inr(sgst)),
                ("Round Off", fmt_inr(round_off)),
                ("Grand Total (Rs.)", fmt_inr(grand)),
            ]
        )
    )
    story += [Spacer(1, 14), Paragraph("E. &amp; O.E. Subject to Nagpur jurisdiction.", SMALL)]

    doc = make_doc(out / "03_multipage_roundoff.pdf", "Tax Invoice GEH/1124/2025")
    footer = page_footer("Ganga Electricals & Hardware | GEH/1124/2025")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------------------
# 04: classic layout, rasterized into an image-only "scan"
# --------------------------------------------------------------------------------------


def make_04(out: Path) -> None:
    seller = PartyInfo(
        "Yamuna Traders",
        ["B-14, Okhla Industrial Area Phase II", "New Delhi, Delhi 110020"],
        "GSTIN",
        gstin("07", "AAHFY2206P"),
    )
    buyer = PartyInfo(
        "Lotus Interiors",
        ["C-5, Sector 18", "Noida, Uttar Pradesh 201301"],
        "GSTIN",
        gstin("09", "ABCFL6640R"),
    )
    items = [
        item("Commercial Plywood 18mm 8x4 ft", "4412", 25, "2150.00"),
        item("Laminate Sheet 1mm Walnut Matte", "4823", 18, "1140.00"),
        item("Edge Banding Tape 22mm (50 m roll)", "3921", 10, "480.00"),
        item("Wood Adhesive 5 kg", "3506", 6, "1325.00"),
    ]
    buf = io.BytesIO()
    build_classic_gst(
        buf,
        seller=seller,
        buyer=buyer,
        meta=[
            ("Invoice No.", "YT/2025/1188"),
            ("Invoice Date", "03/11/2025"),
            ("Due Date", "18/11/2025"),
            ("Place of Supply", "Uttar Pradesh (09)"),
        ],
        items=items,
        taxes=[("IGST @ 18%", "18")],
        footer="Yamuna Traders, New Delhi",
    )

    with pdfplumber.open(io.BytesIO(buf.getvalue())) as pdf:
        img = pdf.pages[0].to_image(resolution=200).original.convert("L")

    rng = random.Random(4)
    img = img.point(lambda p: min(p, 236))  # paper isn't pure white
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(2500):  # dust and toner specks
        x, y, r = rng.randrange(w), rng.randrange(h), rng.choice([0, 0, 1])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rng.randint(60, 170))
    img = img.rotate(1.5, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=228)
    img = img.filter(ImageFilter.GaussianBlur(0.6))

    jpeg = io.BytesIO()
    img.save(jpeg, "JPEG", quality=55)
    scanned = Image.open(io.BytesIO(jpeg.getvalue()))
    scanned.save(
        out / "04_scanned_rotated.pdf",
        "PDF",
        resolution=200.0,
        title="Scanned invoice",
        author="Invoice Intelligence fixture generator",
        creationDate=FIXED_DATE.timetuple(),  # Pillow's PDF writer only accepts struct_time
        modDate=FIXED_DATE.timetuple(),
    )


# --------------------------------------------------------------------------------------
# 05: EU VAT invoice with deliberate defects
# --------------------------------------------------------------------------------------


def make_05(out: Path) -> None:
    seller = PartyInfo(
        "Müller Bürobedarf GmbH",
        ["Friedrichstraße 112", "10117 Berlin, Germany"],
        "VAT ID",
        "DE12345678",  # DEFECT: German VAT IDs have 9 digits after "DE"
    )
    buyer = PartyInfo(
        "Van Dijk Design B.V.",
        ["Keizersgracht 221", "1016 DV Amsterdam, Netherlands"],
        "VAT ID",
        "NL853746219B01",
    )
    items = [
        item("Desk Organizer Bamboo", "BO-1101", 15, "24.90"),
        item("Monitor Stand Aluminium", "BO-2240", 8, "59.00"),
        item("Document Tray A4 Black", "BO-0315", 30, "7.45"),
        item("Paper Shredder P-4 Cross Cut", "BO-5502", 2, "189.00"),
    ]
    printed_net = sum((i.amount for i in items), Decimal("0")) + Decimal("10.00")  # DEFECT: items don't sum to this
    vat = pct(printed_net, "19")
    total = printed_net + vat

    # DEFECTS: no invoice number anywhere; no due date (only "14 days net", which must NOT be turned into a date).
    story: list = [Paragraph(seller.name, H_TITLE)]
    story.append(Paragraph("Friedrichstraße 112 · 10117 Berlin · Germany · VAT ID: " + seller.tax_id, SMALL))
    story.append(Spacer(1, 16))
    story += party_block("Invoice to", buyer)
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Invoice</b>", ParagraphStyle("h", parent=BODY, fontSize=14, leading=18)))
    story.append(Paragraph("Invoice date: 04.11.2025 &nbsp;&nbsp;&nbsp; Delivery date: 31.10.2025", BODY))
    story.append(Spacer(1, 10))

    rows = [["Pos.", "Art. No.", "Description", "Qty", "Unit price", "Total"]]
    rows += [[str(n), i.code, i.description, fmt_qty(i.qty), f"{fmt_eu(i.rate)} €", f"{fmt_eu(i.amount)} €"] for n, i in enumerate(items, 1)]
    t = Table(rows, colWidths=[30, 60, 200, 40, 85, 95])
    style = grid_style("#EFEFEF")
    t.setStyle(style)
    story += [t, Spacer(1, 8)]
    story.append(
        totals_table(
            [
                ("Net total", f"{fmt_eu(printed_net)} €"),
                ("VAT 19%", f"{fmt_eu(vat)} €"),
                ("Total due", f"{fmt_eu(total)} €"),
            ]
        )
    )
    story += [
        Spacer(1, 16),
        Paragraph("Payment terms: 14 days net. Intra-community supply.", BODY),
        Paragraph("IBAN DE89 3704 0044 0532 0130 00 · BIC COBADEFFXXX", SMALL),
    ]

    doc = make_doc(out / "05_eu_vat_defective.pdf", "Invoice")
    footer = page_footer("Müller Bürobedarf GmbH · Amtsgericht Charlottenburg HRB 000000")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


GENERATORS = [make_01, make_02, make_03, make_04, make_05]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for make in GENERATORS:
        make(args.out)
    for pdf in sorted(args.out.glob("*.pdf")):
        print(f"wrote {pdf.relative_to(ROOT) if pdf.is_relative_to(ROOT) else pdf}")


if __name__ == "__main__":
    main()
