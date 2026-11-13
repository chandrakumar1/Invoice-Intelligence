<!--
What:     Explains the images in docs/ (PNGs can't carry a header comment) and how to regenerate them.
In:       Read by anyone updating the README screenshots.
Out:      Screenshots that show real pipeline output, never a mock.
Fails if: The images are regenerated from a mocked result or a different run than the one the README describes.
-->

# docs/

| File | What it shows | Source |
|---|---|---|
| `before.png` | Fixture 05 (`tests/fixtures/pdfs/05_eu_vat_defective.pdf`), page 1, rendered at 120 dpi and cropped above the footer | The PDF itself |
| `after.png` | The review page for that invoice, light theme, 1280 px wide, cropped after the first line item | The stored result of the live accuracy run `data/eval/20261113T102446Z/results/05_eu_vat_defective.json` (Gemini `gemini-3.5-flash-lite`) |

## Regenerating

1. Run the accuracy harness: `.venv/bin/python scripts/evaluate.py`.
2. Copy the result for 05 into a fresh storage folder as `results/<its id>.json`. Serve it with
   `create_app(storage=Storage(folder), llm_factory=<a function that raises>)`, so no LLM call can happen.
3. Screenshot `/?id=<its id>` in a browser at 1280 px, light theme. Headless Chrome works:
   `--headless=new --blink-settings=preferredColorScheme=1 --virtual-time-budget=4000 --window-size=1280,1560`.
4. Render the PDF page with pdfplumber (`page.to_image(resolution=120)`).

Update the README's caption (model, date) if the run changes.
