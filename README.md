# Assignment 4 — Dynamic Knowledge Updating in RAG

Jarvis Team · SE-2418. This study tests whether refreshing the knowledge base
improves the accuracy of release-tag answers in a fixed RAG pipeline. The
frozen evaluation contains 34 GitHub release questions. Each was answered
once in each of four conditions, for a total of 136 predictions.

The four conditions are TF-IDF with the frozen snapshot, TF-IDF with the updated
snapshot, Qwen without retrieval, and BM25 with the updated snapshot. Both
TF-IDF conditions use the retrieval and prompt code from Assignment 3. BM25
uses the same tokenizer, top three records, and retrieval prompt. It uses one
contribution per distinct query term, positive IDF, k1 = 1.2, b = 0.75, and
document ID to break ties.

All conditions use the pinned Qwen2.5-0.5B-Instruct revision on CPU with four
threads, float32, and greedy decoding. The runner shuffles questions with seed
42 and rotates the condition order across questions. Timing starts before
retrieval and ends after answer decoding.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/` | Inference, scoring and plot scripts |
| `data/holdout/` | Frozen questions, reference tags, two knowledge-base snapshots, dataset manifest and collection records |
| `data/pilot/` | The 13 Assignment 3 questions used for the pilot |
| `results/final/` | Original per-question predictions, run metadata and console log |
| `results/final_analysis/` | Original metrics, scored predictions and error cases |
| `results/pilot/`, `results/pilot_analysis/` | Saved pilot evidence |
| `figures/` | The two report figures |

The holdout uses public [GitHub Releases metadata](https://docs.github.com/en/rest/releases/releases#list-releases).
Its two cutoffs are 1 January and 1 June 2025 at 00:00 UTC. The manifest records
SHA-256 digests. `answer_key.json` is read by the scorer and kept separate from
the inputs read by the model. See `data/holdout/README.txt` for provenance and
offline verification.

## Rebuild the results and figures from the original predictions

From the repository root, use Python 3.12. Scoring uses the standard library.
The plotting dependencies are separate from the inference dependencies.

```text
python -m pip install -r requirements-plot.txt
python src/analyze_results.py --dataset holdout --data-dir data/holdout --run results/final --output results/recheck
python src/plot_results.py --summary results/recheck/summary.json --output results/recheck/figures
```

Expected counts are 5/34 for frozen TF-IDF, 34/34 for updated TF-IDF, 0/34 for
Qwen without retrieval, and 34/34 for updated BM25. The original summary has
accuracy, 95% Wilson intervals, retrieval and latency metrics, and the paired
comparisons. `error_cases.csv` records the 29 outdated tags and 34 abstentions.
Choose a new output directory for another scoring run.

## Set up inference and run the Assignment 3 pilot

Use Python 3.12. On Windows, open PowerShell in the extracted project directory.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/run_experiment.py --dataset pilot --output results/pilot_repeat
.\.venv\Scripts\python.exe src/analyze_results.py --dataset pilot --run results/pilot_repeat --output results/pilot_repeat_analysis
```

On macOS, create the environment with `python3.12 -m venv .venv`, then use
`.venv/bin/python` and install the dependencies directly from `requirements.txt`.
The first run downloads the [pinned Qwen snapshot](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
into `.model-cache`. To use a local copy of the exact revision, pass
`--model-dir PATH`. The runner verifies all five model file hashes.

The bundled pilot contains the 13 Assignment 3 questions, including three
development questions and ten evaluation questions. The pilot evidence is
saved in `results/pilot` and `results/pilot_analysis`. Choose fresh output
directories when repeating a run.

## Run the holdout after Appendix A is uploaded

Use the included `data/holdout` directory with the inference environment set up
above. These commands save the repeat separately from the original evidence.

```powershell
.\.venv\Scripts\python.exe src/run_experiment.py --dataset holdout --data-dir data/holdout --output results/reproduction
.\.venv\Scripts\python.exe src/analyze_results.py --dataset holdout --data-dir data/holdout --run results/reproduction --output results/reproduction_analysis
```

The inference command reads the questions and the two snapshots. The scoring
command loads the separate answer key and requires a complete set of paired
predictions. The final dataset manifest is checked against Appendix A. Run
metadata records file hashes, settings, and environment versions. Keep any
failed run directory alongside later attempts. Accuracy and paired metrics
should match the original run. Wall-clock timings vary across runs.
