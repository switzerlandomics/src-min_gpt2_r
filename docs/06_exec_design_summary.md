# Executive Summary

We will build a small **GPT-2–style Transformer** from first principles in R, using a tiny Shakespeare corpus. This model uses a **causal, decoder-only Transformer** architecture: a learned token+position embedding layer, followed by several stacked Transformer blocks (each with multi-head self-attention and a feed-forward network), and a final tied‐embedding softmax to predict the next token. Training is unsupervised (next-token prediction) and the model uses byte-level BPE tokenization with a fixed vocabulary (for example, GPT-2’s released model uses 50,257 tokens and 1024-token contexts). 

We define clear function APIs and tensor shapes (batch×time×embed), separate **core model code** from **infrastructure**, and include optional diagnostic hooks and ggplot2 visualizations for inspection. The project directory is organized into source scripts, data (tokenizer files, sample text), tests, diagnostics and figures. Key milestones include: implementing the forward pass (embedding → attention → feed-forward → logits), verifying gradients via finite differences, overfitting a tiny dataset, and adding generation with key-value caching. All steps and design choices are documented and traced back to the original GPT-2 paper and code.

## Architecture Overview

GPT-2’s architecture is decoder-only and autoregressive.  Each **Transformer block** has two sublayers with *pre*-layer normalization: (1) multi-head self-attention with a causal mask, and (2) a 4×-width feed-forward network with GELU activation.  Formally, for input activations \(X\in\R^{B\times T\times D}\), each block computes  
\[
\begin{aligned}
Y &= X + \mathrm{MultiHeadAttn}(\mathrm{LayerNorm}(X)),\\
Z &= Y + \mathrm{FFN}(\mathrm{LayerNorm}(Y)),
\end{aligned}
\]
where **MultiHeadAttn** splits into \(H\) heads of dimension \(d_h=D/H\), applies scaled-dot-product attention with a causal mask (zeroing out future positions), and merges the heads back to \(D\) dims.  The original “small” GPT-2 had \(D=768\), \(H=12\), \(L=12\) layers, but we will use smaller values for practical reasons.  The final output of the last block is layer-normalized and projected into the vocabulary using the *transpose* of the input embedding matrix (weight tying).  A softmax over vocabulary logits gives the next-token probabilities. Figure below sketches the module hierarchy:

```mermaid
flowchart TD
    A[Scaled Dot-Product Attn] --> B[Attention Head (size $d_h$)]
    B --> C[Multi-Head Attn (concatenate $H$ heads)]
    C --> D[Transformer Block (Attn + FFN + residuals)]
    D --> E[GPT-2 Model (stack $L$ blocks, final LN + output)]
```

Each data tensor has shape **\(B\times T\times D\)** (batch, time, embedding).  The key-value cache (for fast generation) holds tensors of shape \([B, L, 2, H, T_{\text{past}}, d_h]\), where “2” indexes keys/values and \(T_{\text{past}}\) grows during sampling.  

## Required Functions and Operations

We identify the following core functions, with recommended signatures and shapes (\(B\)=batch, \(T\)=sequence length, \(D\)=model dimension, \(V\)=vocab size, \(H\)=#heads, \(d_h=D/H\)):

- **Tokenizer (Byte-Pair Encoding)**  
  - `encode_text(text: string) -> int[B']` – splits a string into BPE token IDs.  
  - `decode_tokens(ids: int[B']) -> string` – maps token IDs back to text.  
  *(These use a provided `vocab.json` and `merges.txt` for ranked merges.)*

- **Embedding and Positional Encoding**  
  - `embed_tokens(X: int[B,T], W_te: float[V,D]) -> float[B,T,D]` – look up token embeddings (gather rows of \(W_{\text{te}}\)).  
  - `add_positional_embedding(H: float[B,T,D], W_pe: float[N_{\text{ctx}},D]) -> float[B,T,D]` – add learned position embeddings (gather from \(W_{\text{pe}}\)).  

- **Combined QKV Projection**  
  - `linear_proj(X: float[B,T,D], W: float[D,3D]+b:float[3D]) -> float[B,T,3D]` – one conv1d-like projection that outputs concatenated \([Q,K,V]\) vectors.  

- **Head Splitting/Merging**  
  - `split_heads(Z: float[B,T,D], H: int) -> float[B,H,T,d_h]` – reshape and transpose to separate heads.  
  - `merge_heads(Z: float[B,H,T,d_h]) -> float[B,T,D]` – inverse of split (transpose+reshape).  

- **Scaled Dot-Product Attention (one head)**  
  - `scaled_dot_attn(Q,K,V: float[B,T,d_h]) -> float[B,T,d_h]` – compute \(W=QK^T/\sqrt{d_h}\), apply causal mask (zero-out future), then softmax and multiply by \(V\).  

- **Multi-Head Attention (module)**  
  - `multi_head_attn(X: float[B,T,D], W_c_attn: float[D,3D], W_c_proj: float[D,D], past: optional float[B,L,2,H,T_p,d_h]) -> (float[B,T,D], present)` –  
    1. Project \(X\to [Q,K,V]\), split into heads, concatenate with any cached past keys/values.  
    2. Apply scaled dot-product per head and softmax.  
    3. Merge heads and apply output projection \(W_{\text{c_proj}}\).  
    4. Return output and updated `present` keys/values.

- **Layer Normalization**  
  - `layer_norm(X: float[B,T,D], γ: float[D], β: float[D]) -> float[B,T,D]` – normalize each vector (zero mean, unit variance) then scale and shift.  Also its backward pass.  

- **Feed-Forward Network**  
  - `ffn(X: float[B,T,D], W1: float[D,4D], b1: float[4D], W2: float[4D,D], b2: float[D]) -> float[B,T,D]` – apply \( \mathrm{GELU}(XW_1 + b_1)\,W_2 + b_2\).  (Use the tanh-based GELU from the code.)  Include backward.  

- **Residual Connections** (handled in block loop) – elementwise add.  

- **Final Output Projection**  
  - `project_output(H: float[B,T,D], W_te: float[V,D]) -> float[B,T,V]` – flatten \(H\) and multiply by \(W_{\text{te}}^T\) to produce logits.  

- **Softmax / Log-Softmax and Loss**  
  - `log_softmax(logits: float[B,T,V]) -> float[B,T,V]`;  
  - `cross_entropy_loss(logits, targets: int[B,T]) -> float` – compute per-token negative log-likelihood and average.

- **Backward-Pass Functions** – one for each forward op above (embedding, linear, split/merge, attn, softmax, etc). These compute gradients of inputs and weights.

- **Optimizer (Adam)**  
  - `adam_update(params, grads, state, lr, β1, β2, ϵ) -> updated_params, updated_state` – classic Adam parameter update.

- **Generation Loop**  
  - `generate(model, prompt_ids, max_len, temperature, top_k, top_p, use_cache)` – repeatedly forward-pass the model, optionally using the key-value cache, to sample or greedily select next tokens.

- **Diagnostic Hooks**  
  - Insert optional callbacks (e.g. `if (inspect) record(tensor, name)`) after key operations: embeddings, QKV split, attention scores, head outputs, residual sums, and final logits. These enable plotting intermediate matrices (e.g. attention heatmaps) without altering core computation.

The table below summarizes tensor shapes in the forward pass (for one batch of size \(B\)):  

| **Operation**               | **Input shape**         | **Output shape**        |
|-----------------------------|-------------------------|-------------------------|
| Token IDs `X`               | \((B, T)\)              | —                       |
| Embeddings + Positions      | \((B, T)\rightarrow(B,T,D)\) | \((B,T,D)\)             |
| QKV projection              | \((B,T,D)\rightarrow(B,T,3D)\) | \((B,T,3D)\)           |
| Split into \(H\) heads      | \((B,T,3D)\rightarrow(B,H,T,3d_h)\) | \((B,H,T,3d_h)\)  |
| Separate Q, K, V            | \((B,H,T,3d_h)\rightarrow(B,H,T,d_h)\times3\) | \(Q,K,V: (B,H,T,d_h)\) |
| Scaled dot-product per head | \((Q,K)\to W\) \((B,H,T,T)\)    | \((B,H,T,T)\)        |
| Attention weights (softmax) | \((B,H,T,T)\)           | \((B,H,T,T)\)           |
| Weighted values per head    | \((B,H,T,T)\times V\to A\) \((B,H,T,d_h)\) | \((B,H,T,d_h)\)   |
| Merge heads + linear        | \((B,H,T,d_h)\rightarrow(B,T,D)\) | \((B,T,D)\)            |
| FFN first linear            | \((B,T,D)\rightarrow(B,T,4D)\) | \((B,T,4D)\)            |
| FFN second linear           | \((B,T,4D)\rightarrow(B,T,D)\) | \((B,T,D)\)            |
| Final logits                | \((B,T,D)\rightarrow(B,T,V)\) | \((B,T,V)\)            |

## Project Structure

We suggest the following directory layout and files:

```
gpt2-r-project/
├── R/                          # R source code
│   ├── tokenizer.R            # BPE tokenizer encode/decode
│   ├── model.R                # Defines Transformer layers (embedding, attn, block)
│   ├── train.R                # Training loop, optimization, logging
│   ├── sample.R               # Inference/generation loop (with & without KV cache)
│   ├── utils.R                # Utilities (tensor ops, losses, gradient checks)
│   └── diagnostics.R         # Optional hooks for capturing tensors
├── data/                      # Data and auxiliary files
│   ├── shakespeare.txt       # Example text corpus
│   ├── vocab.json            # BPE vocabulary (token->ID map)
│   └── merges.txt            # Ranked BPE merges list
├── checkpoints/              # Saved model states (e.g. RDS files)
│   └── example_checkpoint.rds
├── tests/                    # Unit tests
│   ├── test_shapes.R         # Verify tensor shapes flow correctly
│   ├── test_gradients.R      # Finite-difference checks of gradients
│   └── test_overfit.R        # Train on tiny data and check loss goes down
├── figures/                  # Saved diagnostic plots
│   └── (e.g. attn_heatmap.png, ffn_visualization.png, etc.)
└── README.md                 # Project overview, instructions, milestones
```

**Main files and purpose:**  
- `tokenizer.R`: Implements byte-level BPE **encode** and **decode** (using `vocab.json` and `merges.txt`).  
- `model.R`: Defines the Transformer model: embedding lookups, positional addition, combined QKV projection, split/merge, attention, layer-norm, feed-forward, and final output projection (no training loop here).  
- `utils.R`: Contains common routines (e.g. linear layers, layer-norm forward/backward, softmax, cross-entropy, Adam optimizer, numerical gradient checker).  
- `train.R`: Loads data, initializes parameters, iterates forward/backward updates, logs losses, and saves checkpoints.  
- `sample.R`: Generates text from a prompt by running the model forward step-by-step; implements both naive recompute and fast cached modes (matching GPT-2’s `past` mechanism).  
- `diagnostics.R`: Provides optional instrumentation (e.g. capturing intermediate arrays into data frames, calling `ggplot2` to produce sanity-check plots like attention heatmaps or weight histograms).

**Supplementary files:**  
- **Tokenizer files**: `vocab.json` (string-to-ID map of ~50k tokens) and `merges.txt` (Ordered BPE merge rules) allow reproducing GPT-2’s byte-level tokenization.  
- **Shakespeare sample**: A small text file (`shakespeare.txt`) for quick experiments and overfitting tests.  
- **Checkpoint format**: Example RDS or serialized list of weight matrices (`example_checkpoint.rds`), showing how to save/load model parameters (embedding matrices, layer weights, etc.).  

## Visualizations and Diagnostics

Optional diagnostics are crucial for understanding and verifying the model. We will embed **hooks** in the forward pass to capture intermediate results (without disturbing the math). For example: after computing the attention scores \(QK^T\) or after softmax, store these in a list for later plotting. Useful plots include: 

- **Attention heatmaps:** For a small input (e.g. “To be”), plot each head’s attention matrix \(\mathrm{softmax}(QK^T/\sqrt{d_h})\).  
- **Head outputs:** Show how each head’s weighted sum differs across heads.  
- **Residual/activation flow:** Plot how a sample token’s representation changes after each block.  
- **Embedding & positional norms:** Scatterplots or histograms of embedding vector components.  
- **Training curves:** Loss vs iterations (training/validation).  

All plotting can use **ggplot2**. For clarity, each figure must state whether it’s a **schematic** or actual model measurement. Diagnostic code should be guarded (e.g. `if (capture) { ... }`) so that normal training is not slowed by recording.  

## README Outline (Draft)

- **Project**: *GPT-2 from scratch in R*. Builds a toy GPT-2–style Transformer for character/word prediction.  
- **Dependencies**: R (>=4.x), `data.table`, `dplyr`, `ggplot2`, etc. (no deep-learning library needed).  
- **Setup**: Clone repo, install R packages, prepare data (see `data/`).  
- **Usage**:  
  1. **Train**: Run `Rscript R/train.R` with default config (small model, Shakespeare data). Training logs show loss; checkpoints saved.  
  2. **Test**: Use `tests/` scripts to verify shapes and gradients.  
  3. **Generate**: `Rscript R/sample.R --prompt "To be or not"` to sample continuations.  
- **Milestones**:  
  - *Architecture* – implement forward pass (tokenize → embedding → Transformer blocks → output) and verify dimensions step-by-step.  
  - *Gradients* – implement and numerically check backward pass for each module (use small random model, tiny data).  
  - *Training* – integrate optimizer (Adam) and demonstrate overfitting a mini-corpus (loss → 0).  
  - *Generation* – add autoregressive sampling with causal mask and key-value caching, verify cached vs uncached consistency.  
  - *Visualization* – generate attention/activation plots to illustrate internal workings.  
- **Reproducibility**: Set random seeds for R and ensure deterministic tokenizer (the BPE merge list is fixed). Training scripts log the environment (R version, seed) and can be rerun.  
- **Notes**: This is an educational implementation; it does **not** aim to match GPT-2’s scale or performance. We use a very small model (e.g. \(L=2\), \(D=128\), \(H=4\)) for speed. Code is written for clarity, not efficiency.  

## Core vs. Infrastructure Components

We distinguish **core components** (the model’s mathematical operations) from **supporting infrastructure** (scaling, I/O, utilities):

| Component                      | Type             | Notes                                   |
|--------------------------------|------------------|-----------------------------------------|
| Tokenizer (BPE encode/decode)  | Core             | Essential for mapping text↔tokens (data preprocessing). |
| Embedding layers (`W_te`, `W_pe`)| Core           | Learned parameters of the model.        |
| Multi-head Attention (QKV, mask)| Core           | Central computation (causal self-attention).   |
| Feed-Forward Net (2-layer, GELU)| Core           | Position-wise MLP with GELU.     |
| Layer Normalization            | Core             | Applied before each sublayer.   |
| Residual connections           | Core             | Simple adds in forward pass.            |
| Output projection (tied)       | Core             | Final logits via \(H W_{\text{te}}^T\). |
| Loss & Backprop                | Core             | Manual computation of gradients for all ops. |
| Adam Optimizer                 | Infrastructure   | Algorithmic part of training loop.      |
| Data batching/shuffling        | Infrastructure   | Prepares mini-batches, not part of model math. |
| Checkpoint I/O                 | Infrastructure   | Saving/loading parameters.              |
| GPU/parallel code optimizations| Infrastructure   | *Not needed for a small educational model.* |

**Core** = everything implemented “by hand” to learn the math. **Infrastructure** = useful but not mathematically novel (data loaders, training loop, etc.).

## Testing and Validation Checklist

We will write unit tests and checks for each component, for example:

- **Shape Checks**: Given toy inputs, verify each function’s output dimensions. E.g. splitting heads: input \((B,T,D)\) → output \((B,H,T,d_h)\).  
- **Tokeniser Round-Trip**: `decode(encode(text)) == text` for sample strings (ensures BPE is implemented correctly).  
- **Attention Mask Correctness**: Confirm causal mask matrix is lower-triangular (zeros above diagonal) for a given \(T\).  
- **Softmax/LogSoftmax**: Check that `log_softmax` output sums to 1 and equals `softmax` plus log.  
- **Analytic vs Numeric Gradients**: For each custom op (linear layer, split/merge, layer-norm, GELU, etc.), compute gradients by finite differences on a small input and compare to hand-coded backprop (within tolerance).  
- **One-Step Generation Consistency**: Compare logits from “fresh compute” vs “cache-enabled” generation for the same prompt (they should match up to numerical precision).  
- **Overfit Tiny Dataset**: Train the model on a very short fixed sequence (e.g. “shakespeare.txt”) for enough epochs and verify training loss → near 0 (model can memorize).  
- **Reproducibility**: Fix RNG seed; check that forward pass (and gradients) produce the same results on repeated runs.  

Each test script should output clear PASS/FAIL messages or use an R testing framework (e.g. `testthat`). This ensures correctness of shapes and learning before proceeding to larger experiments.  

## References

We follow the official GPT-2 design and code (OpenAI, 2019) as primary guides. In particular, we refer to the GPT-2 repository for exact operations (e.g. the combined QKV projection and head splitting in `attn`, and the use of tied weights for the output projection) and the Hugging Face model card for key architecture notes (byte-level BPE tokenization with 50,257 tokens and 1024-length context, causal attention masking). All design choices in this spec are traceable to these sources. 


Yes. The plan is technically sound, but I would reduce its initial scope. Your objective is to understand and verify GPT-2’s computations in R, not to build a general-purpose training framework. The original OpenAI implementation supports a relatively compact design: embeddings, repeated pre-normalised Transformer blocks, multi-head causal attention, a GELU feed-forward network, final layer normalisation and a tied output projection. We should keep that computation at the centre of the repository.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

+1

I would make five adjustments before treating the plan as our working specification.

|
Area

|

Keep or change

|

Reason

|
| --- | --- | --- |
|

Tokenizer

|

Start with a simple byte-token baseline; add byte-level BPE as the next isolated component.

|

This lets us test the model independently of tokenizer bugs. BPE remains part of the completed project, but we do not need to implement vocabulary training before testing attention and gradients.

|
|

Model

|

Implement two blocks and multiple heads, with configurable dimensions.

|

This introduces the important GPT-2 mechanisms without requiring its original parameter count or context length.

|
|

Training

|

Keep manual backpropagation, gradient checks, one straightforward optimiser and a small training loop.

|

These are essential for understanding whether our implementation actually learns. Do not add multiple optimiser variants or an elaborate scheduling system initially.

|
|

Generation

|

Implement ordinary autoregressive generation first. Defer key-value caching.

|

Caching reuses previous attention computations; it does not change the underlying model. We can add and verify it once ordinary generation works.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

+1

|
|

Diagnostics

|

Reserve a clean interface for optional inspection, but initially implement only a few useful outputs.

|

We can capture real attention weights, token boundaries and learning curves without building an extensive visualisation framework before the model is working.

|

Two technical corrections are worth making in the research plan itself. GPT-2 splits the combined QKV projection into Q, K and V before splitting each into heads; our function contracts should reflect that order. Also, causal masking must operate on attention scores before softmax, rather than simply zeroing future scores. The original source makes both operations explicit.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

The testing requirements should also stay precise. For example, `exp(log_softmax(logits))` should sum to one, not `log_softmax(logits)` itself. A tiny-data overfitting test should demonstrate a substantial, reproducible loss reduction; we should not require loss to approach zero when identical contexts may have different targets.

## The lean project structure

I would retain the familiar structure from your previous Transformer and avoid creating a separate file for every mathematical operation.

```
gpt2-r/
├── README.md
├── LICENSE
├── R/
│   ├── tokenizer.R       # Byte tokens; later byte-level BPE
│   ├── data.R            # Corpus preparation, splits and batches
│   ├── model.R           # Parameters, forward pass and manual backward pass
│   ├── optimiser.R       # Adam and gradient clipping
│   ├── evaluation.R      # Loss, validation and generation
│   └── plots.R           # Optional figures from recorded results
├── experiments/
│   └── run.R             # Configure, train, evaluate and save a run
├── tests/
│   ├── test_tokenizer.R
│   ├── test_model.R
│   ├── test_gradients.R
│   └── test_training.R
├── data/
│   └── README.md         # Corpus source and preparation instructions
├── docs/
│   └── design.md         # Architecture, tensor shapes and implementation notes
└── output/               # Generated runs; excluded from Git
```

`R/model.R` should initially contain both forward and backward functions. Keeping their corresponding calculations together will make it easier to inspect the mathematics and trace errors. If that file eventually becomes genuinely difficult to navigate, we can split it by meaningful module boundaries. We do not need to predict that problem in advance.

Use `token_ids`, `targets`, `hidden_states`, `attention_scores`, `attention_weights`, `parameters` and `gradients` as general-purpose names. Shakespeare belongs in a corpus configuration or file path, not in model-function or variable names.

For the first experiment, Tiny Shakespeare remains suitable: it is familiar, small enough for repeated tests and allows a direct continuation of your earlier work. Its limited size means we should start with a modest vocabulary and model, rather than importing GPT-2’s full 50,257-token vocabulary. The original encoder loads an existing BPE vocabulary and ranked merge rules, so training our own small vocabulary is a distinct task we can add once the model is verified.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

+1

My proposed first milestone: a two-block, multi-head model in R that passes deterministic shape and gradient tests, learns from a short token sequence and generates text using ordinary autoregressive sampling. At that point we will have verified the principal architectural step beyond your previous Transformer. We can then add byte-level BPE, longer experiments and selected diagnostics one at a time, only where they teach us something or solve a demonstrated problem.


