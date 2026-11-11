<!--
What:     Project front page for clients and contributors (stub until M7 adds results and screenshots).
In:       Nothing.
Out:      What the project is, how to run it, and why each dependency exists.
Fails if: It claims accuracy the harness hasn't measured. Numbers only arrive with M7.
-->

# Invoice Intelligence

Turns messy invoice PDFs (including scanned ones) into validated JSON where **every field
cites the page and the verbatim text it came from**, so a human can verify it rather than trust it.

> Work in progress. Milestone 1 of 7 (schemas and test fixtures) is complete.
> The problem statement, screenshots, and an honest accuracy table arrive in M7.

## Design in one paragraph
The LLM only **transcribes and points**: for each field it returns the exact characters and the
page/line where they appear. Python does everything else. It slices the real quote from the page
text, verifies the value is actually there, parses amounts and dates, scores confidence, and runs every
arithmetic and format check. A value whose citation can't be verified is dropped (`None` plus a reason),
never guessed.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # add ANTHROPIC_API_KEY (needed from M3)
brew install tesseract          # OCR for scanned pages (needed from M2)
.venv/bin/pytest
```

**Python version:** tested on **Python 3.14.3** (macOS, x86_64). Every dependency installs from
prebuilt wheels with `--only-binary=:all:`, so no compiler is needed and a 3.12 venv isn't required.
Python ≥ 3.12 is the minimum (the code uses PEP 695 generics).

## Dependencies
| Package | Why it's here |
|---|---|
| fastapi | The two HTTP endpoints, with request/response validation from the same Pydantic models |
| uvicorn | ASGI server to run FastAPI |
| python-multipart | Required by FastAPI to accept file uploads |
| pydantic | The schema contract; imported directly, so pinned directly |
| pdfplumber | Text layer with character positions; also renders pages for OCR (via pypdfium2), so there's no poppler |
| pytesseract | Thin wrapper over the Tesseract OCR binary for scanned pages |
| anthropic | Official SDK for tool-use structured extraction, with retries |
| python-dotenv | Loads `.env` so the project runs with a single copy step |
| *dev:* pytest | Test runner |
| *dev:* httpx | Required by FastAPI's `TestClient` |
| *dev:* reportlab | Generates the synthetic fixture PDFs (dev only, pinned so output stays byte-identical) |

Deliberately absent: pandas, OpenCV, rapidfuzz (stdlib `difflib` is enough), any database, any frontend framework.
