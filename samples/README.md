# Sample receipts

A small subset of the [WildReceipt](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)
dataset (Apache-2.0), committed so the app can be tested on any machine — including a fresh
`git clone` on a work laptop — **without downloading the full dataset**.

Try one:

- **Web UI:** `python run_web.py` → open http://127.0.0.1:8000 → drag a `sample_*.jpeg` onto the page.
- **CLI:** `pip install -e .` once, then `slipguard validate samples/sample_01.jpeg --provider groq`

The full WildReceipt set (under the gitignored `datasets/`) is only needed for the `eval-*`
benchmarks, not for testing the app.
