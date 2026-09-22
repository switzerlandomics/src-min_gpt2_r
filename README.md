# GPT-2-style language model from scratch in R

A small, CPU-only **decoder-only Transformer** implemented directly in R. This
project follows our character-level RNN and single-head Transformer, and shows
how multi-head causal attention, independently parameterised stacked blocks,
GELU feed-forward layers and tied input/output embeddings predict the next token.

This is an educational **architectural reproduction**, not OpenAI's original
GPT-2 pretraining pipeline, scale or language quality.

## Scope and dependencies

The model implements its forward pass, complete manual backward pass, next-token
cross-entropy, Adam updates, numerical gradient checks, held-out validation,
checkpointing and uncached autoregressive sampling in ordinary R matrices and
arrays. It currently uses **individual UTF-8 bytes**, with 256 one-based token
IDs. GPT-2's byte-level BPE and key-value-cached generation are future, separate
steps. No deep-learning framework, automatic differentiation, pretrained neural
weights or packaged Transformer layers are used for training.

`ggplot2` is optional for saved figures; `svglite` is optional for editable SVG
publication figures. Neither package is required to train or evaluate the model.
Install the plotting packages only if needed:

```r
install.packages(c("ggplot2", "svglite"))
```

## Quick start

Run these commands from the repository root:

```sh
Rscript tests/test_tokenizer.R
Rscript tests/test_model.R
Rscript tests/test_gradients.R
Rscript tests/test_training.R
Rscript experiments/run.R --iterations=100
```

`data/input.txt` is a small offline smoke-test corpus, not a meaningful
language-modelling benchmark. Supply your own local copy of Tiny Shakespeare
at `data/tiny_shakespeare.txt` for a longer experiment:

```sh
Rscript experiments/run.R \
  --input=data/tiny_shakespeare.txt \
  --iterations=2000 \
  --context-length=32 \
  --embedding-size=32 \
  --n-heads=4 \
  --n-layers=2 \
  --batch-size=2 \
  --validation-interval=100 \
  --checkpoint-interval=500 \
  --plot-detailed
```

Use `Rscript experiments/run.R --help` for the complete set of options.
Training uses the CPU and may be slower than a framework/GPU implementation.
No training duration or generated-text quality target is guaranteed.


## Heaviest runs

The initial 2,000-update experiment completed in approximately 1 minute on one laptop CPU. The following two configurations provide a progression towards longer-context learning and a larger GPT-2-style model. Runtime estimates are approximate, based on that initial run; CPU performance and validation overhead will affect the actual duration.

Intermediate run — aim for approximately 5 minutes. Increase the context from 32 to 48 tokens, the embedding size from 32 to 48, and the number of training updates from 2,000 to 4,000. Keep two blocks, four heads and the same batch size so the architectural changes remain easy to interpret.

<!-- Private example may not be visible to readers (20260922_163712) -->
```sh
Rscript experiments/run.R \
  --input=data/tiny_shakespeare.txt \
  --iterations=4000 \
  --context-length=48 \
  --embedding-size=48 \
  --n-heads=4 \
  --n-layers=2 \
  --batch-size=2 \
  --validation-interval=200 \
  --checkpoint-interval=1000 \
  --plot-detailed
```

Larger run — allow approximately 1–2 hours, potentially longer. Increase the context to 64 tokens, embedding size to 64, depth to three blocks, and batch size to four. Use 10,000 updates to give this larger model more opportunities to learn. Retaining four heads increases each head’s feature dimension from 8 in the original run to 16 here, without changing the number of heads.


<!-- Private example may not be visible to readers (20260922_164139) -->
```sh
Rscript experiments/run.R \
  --input=data/tiny_shakespeare.txt \
  --iterations=10000 \
  --context-length=64 \
  --embedding-size=64 \
  --n-heads=4 \
  --n-layers=3 \
  --batch-size=4 \
  --validation-interval=500 \
  --checkpoint-interval=1000 \
  --plot-detailed
```

The larger run does substantially more computation per update, as well as performing more updates. It should not be expected to finish in five times the duration of the original run simply because it has five times as many updates. Start with the intermediate run to establish a timing reference on your own machine before committing to the larger one.

Both configurations save metrics, a live training figure and resumable checkpoints during training. Detailed attention and prediction figures are generated at the end. These experiments use the same byte-token vocabulary, but changing the context length also changes the validation windows, so their reported validation losses are not a strictly controlled model-size comparison.


## Monitoring and publication figures

`--plot` refreshes `training.png` at every validation measurement while the
experiment continues. `metrics.csv` and resumable checkpoints are saved
independently of plotting. The training series is loss from sampled windows;
validation measures fixed held-out windows. Neither series is smoothed.

`--plot-detailed` **implies `--plot`**. After training it uses the saved
best-validation checkpoint and the same short validation prompt to create:

| Output under `output/YOUR_RUN/detailed/` | Measured content |
|---|---|
| `training_detail.png` and optionally `.svg` | Learning history and selected checkpoint. |
| `next_token_probabilities.png` and optionally `.svg` | Token probabilities from the selected model, compared with the initial model when available. |
| `attention_block_01_head_01.png` and optionally `.svg`, etc. | One compact attention matrix for **every head in every block**. All use the same 0–1 scale; grey cells indicate masked future positions. |

The main output directory also contains a compact `attention.png` for the first
head of the first block, generated at the end of the run. Detailed figures are
not recalculated at every training update. Plotting failures produce warnings,
not training failures. If `svglite` is absent, detailed PNGs are still saved.

The plots use a shared teal, blue-grey and neutral palette and export at
approximately the intended blog presentation size. Theme text is **at least
12 pt in the SVG**, with 16 pt titles. Single-head attention exports at
4.5 × 4.5 inches; next-token predictions at 5 × 4.5 inches; and learning curves
at 5.5 × 4 or 5.5 × 4.5 inches. Reducing an entire SVG in Inkscape or on the
website will also reduce its apparent text size. Use a standard sans-serif font
or an equivalent installed locally; no font files are included.

To produce detailed plots for an already completed run without further updates:

```sh
Rscript experiments/run.R \
  --resume=output/YOUR_RUN \
  --iterations=2000 \
  --plot-detailed
```

Set `--iterations` to that run's saved **total** update count. The selected model
is chosen by validation loss, not by sample readability. Older runs without
`initial_model.rds` show only the selected model's next-token probabilities;
missing initial predictions are not reconstructed.


### From text to training data

The model uses a byte-level tokenizer: each UTF-8 byte becomes one of 256 token IDs (byte value + 1 for R’s one-based indexing). For example, `"you are,"` becomes eight tokens. This is not yet GPT-2’s byte-pair encoding (BPE).

The demonstration script uses the project’s actual tokenizer and batching functions to show how text becomes token IDs, how Tiny Shakespeare is split into training and validation data, and how each training input is paired with its next-byte target.

Bash

```
Rscript experiments/inspect_tokenizer.R \
  --input=data/tiny_shakespeare.txt \
  --text="you are," \
  --context-length=8 \
  --start=1 \
  --output=output/tokenization.md
```

[View the full tokenisation report](/output/tokenization.md) , including the byte-by-byte tables and a real training window: input `"First Ci"` → targets `"irst Cit"`.


## Reproducibility and outputs

Each dated run under `output/` records its original command, resolved options,
corpus checksum, best-validation model, complete resumable checkpoint, metrics
and a model-generated sample. `experiment.log` reproduces the concise console
progress reports with elapsed time and an ETA measured from the current session.
On resume, the corpus checksum is checked, and only the total iteration target
and plotting flags may change. Saving the complete checkpoint at validation
also keeps model-selection metadata and recoverable training history aligned.

```text
output/YOUR_RUN/
├── run_command.txt
├── corpus_checksum.txt
├── experiment.log
├── metrics.csv
├── initial_model.rds
├── best_model.rds
├── latest_checkpoint.rds
├── sample.txt
├── training.png                      # With --plot
├── attention.png                     # With --plot
└── detailed/                         # With --plot-detailed
    ├── training_detail.png            # Plus .svg if svglite is installed
    ├── next_token_probabilities.png   # Plus .svg if svglite is installed
    ├── attention_block_01_head_01.png # Plus .svg if svglite is installed
    └── ...                            # One pair per block and head
```

The test portion of the corpus is held aside and is not used for ordinary
checkpoint selection. The original training corpus remains on your machine.

## Repository structure

```text
R/tokenizer.R           Byte-token encoder and decoder
R/data.R                Raw corpus I/O, splits and batches
R/model.R               Parameters, GPT-2-style forward and manual backward
R/optimiser.R           Adam and global gradient-norm clipping
R/evaluation.R          Held-out loss and autoregressive generation
R/plots.R               Optional measured monitoring and publication figures
R/progress.R            Human-readable logs, session ETA and file-writing helpers
experiments/run.R       Reproducible training, checkpointing and plotting options
tests/                  Tokeniser, shape, gradient and training checks
data/                   Offline example and corpus instructions
docs/                   Model and optional figure-design notes
output/                 Local experiment artefacts, excluded from Git
```

The model's array convention is `[batch, time, features]` and its identifiers
use general names such as `token_ids`, `targets`, `parameters` and `gradients`.
The causal mask is applied to scores **before softmax**; the output projection
shares the token-embedding matrix and accumulates gradients from both uses.

## Primary references

- OpenAI's [original GPT-2 repository](https://github.com/openai/gpt-2),
  particularly [`src/model.py`](https://github.com/openai/gpt-2/blob/master/src/model.py),
  [`src/encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
  and [`src/sample.py`](https://github.com/openai/gpt-2/blob/master/src/sample.py).
- Radford et al. (2019), [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- [Released GPT-2 weights and model card](https://huggingface.co/openai-community/gpt2),
  relevant to optional future compatibility testing, not this project's training.

The R training settings are independently defined and do not claim to recreate
OpenAI's unreleased original GPT-2 training pipeline.
