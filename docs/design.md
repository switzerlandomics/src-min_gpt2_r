# Preliminary implementation design

This design may change after numerical checks and initial training. The aim is
to understand the GPT-2 computation, not to reproduce the original model's size,
pretraining corpus or language quality.

## Shapes and operations

Use R arrays in `[batch, time, features]` order, token IDs and targets as
`[batch, time]` matrices, and one-based token IDs. For efficiency, position-wise
linear algebra flattens batch and time into matrix rows, then restores the
original dimensions. A block's combined QKV projection produces `3 * d`
features. Split that result into Q, K, V first; each is then partitioned across
`n_heads` contiguous feature groups of width `d / n_heads`.

Every head applies `Q %*% t(K) / sqrt(head_size)`, masks future scores with
`-Inf` **before** row softmax, and takes the attention-weighted sum of V.
Concatenate head outputs, apply the attention output projection, and add the
unmodified block input. Apply pre-layer norm, a 4*d-wide feed-forward layer
with the original tanh GELU approximation, project back to d and add the second
residual. Repeat with independent block parameters. Apply final layer norm and
multiply by the transpose of the input token embedding matrix, tying input
and output weights.

## Learning and verification

Train on shifted next-token targets. The manual backward pass propagates through
both residual paths and every block in reverse order, and accumulates the input
embedding gradient with the gradient from the tied output projection. A small,
deterministic finite-difference test checks the implementation; a toy sequence
must show measurable learning before longer experiments. Training and
validation losses are measured in **nats per byte token**, and are not directly
comparable with subword or character vocabulary losses.

The validation split is used for selecting `best_model.rds`. The test split is
not evaluated automatically. Checkpoint state contains model parameters,
optimiser moments, RNG and experiment progress. On resume, the file checksum
is checked to detect changed input. Generated examples are seeded separately
from training and use the best-validation checkpoint.

## Deliberate omissions and later work

This first runnable build uses single raw-byte tokens, not GPT-2's byte-level
BPE, and recomputes context during generation rather than caching keys and
values. A separately tested BPE tokenizer and KV-cache comparison can follow.
The published GPT-2 tokenizer, checkpoints, GPU/distributed/mixed-precision
training and original WebText dataset are not used. The released GPT-2 source
specifies forward and sampling operations, not a complete original pretraining
pipeline, so this project's training schedule is its own documented choice.

## Optional inspection

Normal training does not collect attention plots. `model_forward(..., inspect =
TRUE)` returns measured per-head attention scores/weights and block hidden
states for a small input. `R/plots.R` can produce a measured attention heatmap
and plot recorded training/validation losses. Potential later figures: token
boundaries as BPE merges occur, QKV/head tensor shapes, residual additions,
representations across blocks and token probabilities at fixed checkpoints.
Distinguish schematic illustrations from measurements recorded from a real run.

## Original references

- OpenAI (2019): https://github.com/openai/gpt-2/blob/master/src/model.py
- Original tokenizer: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- Original sampling: https://github.com/openai/gpt-2/blob/master/src/sample.py
- GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Released small model: https://huggingface.co/openai-community/gpt2
