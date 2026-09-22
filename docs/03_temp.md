# GPT-2 from first principles in R: source-code review and implementation strategy

For this project, I would use the original GPT-2 implementation as the architectural specification, the 2019 paper as the historical and experimental reference, and your previous R projects as the starting point for the learning and validation code.

The distinction matters. Many descriptions of “GPT-2 architecture” mix together the original model, later implementations and modern Transformer improvements. For a first-principles reproduction, we should first establish what GPT-2 actually computes, then decide which parts must be reproduced exactly, which can be reduced in size, and which belong to the surrounding training infrastructure.

I reviewed OpenAI’s original [`model.py`](https://github.com/openai/gpt-2/blob/master/src/model.py) , [`encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py) , [`sample.py`](https://github.com/openai/gpt-2/blob/master/src/sample.py)  and the paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language-models.pdf) . These provide a particularly useful foundation because the released source is compact enough to inspect function by function.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+3

## 1. First establish what we are reproducing

GPT-2 is a decoder-only, autoregressive Transformer language model. It receives a sequence of token IDs and calculates a probability distribution for the next token at every position. During training, the actual next tokens provide the targets; during generation, the model samples a token and uses it as part of the context for the following prediction. Its attention is causal, so a position cannot access tokens that come after it.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+3

The smallest released GPT-2 configuration provides a useful historical reference:

|
Parameter

|

Original small GPT-2

|
| --- | --- |
|

Vocabulary

|

50,257 tokens

|
|

Maximum context

|

1,024 tokens

|
|

Embedding width

|

768

|
|

Attention heads per block

|

12

|
|

Transformer blocks

|

12

|
|

Feed-forward width

|

3,072

|

These values describe the original small-model configuration, not the dimensions we need for an educational R implementation. The published paper labels its smallest model approximately 117 million parameters; OpenAI’s repository subsequently notes that some original parameter counts were incorrect.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+2

A crucial design principle follows: architecture and scale are separate questions. A model with two blocks, four attention heads and a small vocabulary can implement the same architectural operations without reproducing GPT-2’s parameter count or language capabilities.

## 2. The exact forward pass: from text to next-token probabilities

This is the computation we should be able to trace completely, using a short example, before attempting an extended training run.

GPT-2 computation hierarchy

Input text → token IDs

Token embeddings + positional embeddings

Transformer block × N

Layer norm → multi-head causal attention → residual addition

Layer norm → feed-forward network → residual addition

Final layer norm → vocabulary logits

Next-token probabilities → sampled token

Architectural overview of the forward and generation paths. Training uses logits at all eligible positions; generation uses the final position.

The order above comes directly from the released `model()`, `block()`, `attn()` and `mlp()` functions.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

### 2.1. Text encoding: byte-level BPE

Your character models assign one vocabulary entry to each distinct character. GPT-2 instead uses byte-level byte-pair encoding (BPE), which can represent frequent sequences using larger tokens while retaining byte-level coverage for unfamiliar text.

The original `encoder.py` performs several distinct operations: it divides text into pieces using a regular-expression pattern; converts each piece’s UTF-8 bytes into a reversible Unicode representation; repeatedly applies ranked BPE merges; and maps the resulting pieces to token IDs. Decoding reverses those operations. The encoder loads a vocabulary and merge rules from `encoder.json` and `vocab.bpe`.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+2

For our implementation, these should be separately understandable functions:

```
encode_text()
  ├── split_text()
  ├── utf8_bytes_to_symbols()
  ├── apply_ranked_bpe_merges()
  └── symbols_to_token_ids()

decode_tokens()
  ├── token_ids_to_symbols()
  ├── symbols_to_utf8_bytes()
  └── bytes_to_text()
```

An important distinction for the project: implementing a BPE encoder and decoder is not the same as training the BPE vocabulary. The released GPT-2 encoder loads an existing set of merge rules; it does not provide the complete procedure used to learn those rules from the original training corpus. We should document our own vocabulary-training procedure separately if we choose to build one.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

For learning, I would initially train a small byte-level BPE vocabulary using training text only. That makes the merge decisions inspectable and keeps the output layer affordable in R. We can later test compatibility with the original GPT-2 tokenizer as a separate exercise. Using its published vocabulary would not constitute using pretrained neural-network weights, but it would mean we had not learned the tokenizer ourselves.

### 2.2. Token and positional embeddings

Let the batch contain BBB sequences of TTT token IDs. GPT-2 looks up a learned token embedding and a learned absolute positional embedding for every input position, then adds them:

Xb,t,:=Etoken\(xb,t,:\)+Eposition\(t,:\).X_{b,t,:}=E_{\mathrm{token}}\(x\_{b,t},:\)+E_{\mathrm{position}}\(t,:\).Xb,t,:=Etoken\(xb,t​,:\)+Eposition\(t,:\).

The resulting activation tensor has shape B×T×dB\times T\times dB×T×d, where ddd is the embedding width. Unlike the original 2017 Transformer’s principal sinusoidal-position configuration, GPT-2 uses learned positional embeddings. Its published implementation initialises and adds the token and positional embeddings in `model()`.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

This is already familiar from your first Transformer. The next project should add explicit handling of the batch dimension and establish a consistent tensor-shape convention throughout the R code.

### 2.3. Layer normalisation before attention

Each Transformer block begins by layer-normalising its input before the attention sublayer. GPT-2 uses learned scale and bias parameters and an epsilon of 10−510^{-5}10−5 in the released implementation. The normalisation is across the feature dimension of each position, not across the positions of the sequence.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

This pre-normalisation is not a new concept for your project: your minimal Transformer already uses it. The new issue is ensuring that the same calculation and its gradients work correctly for a batched, multilayer model.

### 2.4. The combined QKV projection

This is the first substantial architectural change.

Your initial model constructs one set of queries, keys and values for one attention head. GPT-2’s `attn()` function performs one combined learned projection whose output width is 3d3d3d, then splits that result into QQQ, KKK and VVV. Each is subsequently reshaped into multiple heads.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

For HHH attention heads, the head dimension is

dh=dH.d_h=\frac{d}{H}.dh=Hd.

Each head receives its own dhd_hdh-dimensional query, key and value vectors. A useful working shape convention is:

|
Quantity

|

Shape

|
| --- | --- |
|

Block input

|

B×T×dB\times T\times dB×T×d

|
|

Combined QKV projection

|

B×T×3dB\times T\times 3dB×T×3d

|
|

Q, K or V after splitting

|

B×H×T×dhB\times H\times T\times d_hB×H×T×dh

|
|

Attention scores

|

B×H×T×TB\times H\times T\times TB×H×T×T

|

The original code explicitly asserts that embedding width is divisible by the number of heads and uses `split_heads()` and `merge_heads()` to perform the reshaping.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

For readability, I would expose Q, K and V as separately named intermediate arrays in our diagnostic mode, even if the computational implementation uses one combined projection.

### 2.5. Scaled dot-product attention inside every head

Each head independently calculates scaled query–key scores, applies the causal mask, normalises the permitted scores using softmax and computes a weighted sum of its value vectors:

Ah=softmax⁡ ⁣(QhKhTdh+M)Vh.A_h= \operatorname{softmax}\!\left( \frac{Q_hK_h^{\mathsf T}}{\sqrt{d_h}}+M \right)V_h.Ah=softmax(dhQhKhT+M)Vh.

There are two details worth reproducing accurately.

First, the scale is the square root of the head dimension, not the full embedding width. Second, masking happens before softmax, so future positions receive zero attention weight in the resulting probability matrix. OpenAI’s original implementation represents masked scores using a very large negative value rather than a literal negative infinity.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

We should retain the matrix-level visualisation approach from your previous project, but generate one attention matrix per head from a recorded model checkpoint. This will let readers see that the heads receive different learned projections and can produce different attention patterns.

A caution for the figures: a higher attention weight is a measurable coefficient in this calculation, not by itself a complete explanation of why the network made a prediction.

### 2.6. Concatenating heads and projecting their output

After each head has produced its output, GPT-2 merges the head outputs back into the full embedding width and applies an additional learned linear projection, named `c_proj` in the original attention module.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

This is where the hierarchy becomes particularly clear:

```
Scaled dot-product attention
         ↓
     One head
         ↓
  Multiple heads
         ↓
  Concatenate / merge
         ↓
  Output projection
         ↓
Multi-head attention module
```

For the new article, I would create an actual tensor-shape figure illustrating this transformation. Unlike a decorative network diagram, it would be derived from the dimensions used in the R implementation.

### 2.7. First residual connection

The result of the attention module is added to the block’s input:

R=X+MultiHeadAttention⁡(LN⁡1(X)).R=X+\operatorname{MultiHeadAttention} \bigl(\operatorname{LN}_1(X)\bigr).R=X+MultiHeadAttention(LN1(X)).

The normalised input is used inside attention, while the residual addition uses the original block input. This ordering is explicit in the original `block()` function.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

It is worth keeping the corresponding intermediate arrays separately inspectable. Residual connections can otherwise become difficult to follow once the code contains several stacked blocks.

### 2.8. Feed-forward network with GELU

GPT-2 then layer-normalises the residual representation and applies a position-wise feed-forward network. Its intermediate width is four times the embedding width, and its activation is GELU rather than the ReLU used in your first Transformer. The original source implements the tanh approximation to GELU.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

Using the row-vector convention from your previous R project, the computation can be written as

F(R)=GELU⁡(LN⁡2(R)W1+b1)W2+b2,F(R)= \operatorname{GELU} \bigl(\operatorname{LN}_2(R)W_1+b_1\bigr) W_2+b_2,F(R)=GELU(LN2(R)W1+b1)W2+b2,

followed by the second residual connection:

Y=R+F(R).Y=R+F(R).Y=R+F(R).

This sublayer processes each sequence position independently. The interaction between positions occurs in attention; the feed-forward network transforms each position’s resulting representation.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

GELU and its derivative should be implemented and numerically checked, rather than substituting ReLU and still describing the result as a GPT-2 reproduction.

### 2.9. Repeating the Transformer block

The output of the first block becomes the input of the second, and so on. The original `model()` iterates over `n_layer`, applying the same block architecture with a different learned parameter set for each layer.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

This is the second central change from your current project. Multi-head attention introduces parallel submodules within a layer; stacking introduces successive transformations through depth.

The proposed implementation should therefore expose both levels:

```
model_forward()
  └── transformer_block_forward()    # repeated with separate parameters
        ├── layer_norm()
        ├── multi_head_attention()
        │     └── scaled_dot_product_attention()  # once per head
        ├── residual_add()
        ├── layer_norm()
        ├── feed_forward_gelu()
        └── residual_add()
```

The internal representation after each block is an excellent candidate for an optional diagnostic snapshot.

### 2.10. Final normalisation and tied output weights

After the final Transformer block, GPT-2 applies one more layer normalisation, `ln_f`. It then projects each position’s representation to vocabulary logits using the transpose of the token embedding matrix. The output weights are therefore tied to the input token embeddings rather than stored as an independent learned output matrix.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

For an output representation ZZZ,

logits=ZEtokenT.\mathrm{logits}=ZE_{\mathrm{token}}^{\mathsf T}.logits=ZEtokenT.

The result has shape B×T×VB\times T\times VB×T×V, where VVV is vocabulary size.

Weight tying is not simply a memory-saving implementation trick in this project: it changes the parameterisation of the model. During training, the token embedding matrix receives gradient contributions from both its input-lookup use and its output-projection use. Our manual backward pass must accumulate both contributions correctly.

The original `model.py` ends by returning logits. Softmax, selection of the final position and token sampling belong to the subsequent generation procedure.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

## 3. Training: what the released source establishes, and what we must implement

Here we should be especially precise about source provenance.

The original GPT-2 repository provides the model, tokenizer, sampling utilities and pretrained checkpoints; it is not a complete published reproduction of the original pretraining pipeline. Its `model.py` returns logits and does not provide an end-to-end training loop with the original data pipeline, loss reduction, optimiser schedule and checkpoint machinery. Consequently, our R training system should be described as our explicit training implementation of the GPT-2-style model, rather than claiming that every training-engineering choice reproduces OpenAI’s original procedure.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+3

### 3.1. Construct input-target pairs

For a token sequence

\(t1,t2,…,tT+1\),\(t\_1,t\_2,\\ldots,t\_{T+1}\),\(t1​,t2​,…,tT+1​\),

the inputs are the first TTT tokens, and the targets are the following TTT tokens:

x=\(t1,…,tT\),y=\(t2,…,tT+1\).x=\(t\_1,\\ldots,t\_T\), \qquad y=\(t\_2,\\ldots,t\_{T+1}\).x=\(t1​,…,tT​\),y=\(t2​,…,tT+1​\).

The model produces vocabulary logits at every input position, and the target at that position is the actual next token. This is the same fundamental learning objective as your RNN and first Transformer, now applied to subword-token sequences.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+1

### 3.2. Calculate next-token cross-entropy

We should implement a numerically stable log-softmax and calculate the negative log probability of each target token. The training loss is the mean across the eligible positions in the batch.

This produces a critical new measurement convention: loss in nats per token, rather than nats per character. We must not directly compare the resulting figures with the character-level losses from your earlier projects. Tokenisation changes the unit being predicted.

### 3.3. Differentiate the complete model

This will be one of the largest engineering tasks.

Your existing manual backward-pass work provides the starting point, but the new system must propagate gradients through multiple heads, combined QKV projections, tensor reshaping, concatenation, GELU, residual paths, repeated blocks and the tied embedding/output matrix.

I would design the backward functions alongside the forward functions, with explicit cached intermediates and documented input/output shapes. The first milestone should be finite-difference gradient agreement on a tiny deterministic model, not good generated text.

### 3.4. Optimise and initialise parameters

The paper reports a modified initialisation that accounts for residual accumulation with depth. The released `model.py` also specifies concrete initialisers for its learned projections and embeddings. These are relevant reference points, but the paper’s description and the released inference code should not be assumed to constitute a complete, identical training recipe.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+2

Our project should document its own initialisation, optimiser, learning rate, clipping rules and any schedule explicitly. I would retain your hand-coded Adam initially, and only add further training refinements when a measured problem or a clearly defined reproduction goal justifies them.

## 4. Generation: the exact autoregressive loop

The released `sample.py` provides the next part of the specification.

The generation process starts with a prompt or starting token, runs the model, selects logits at the final position, applies temperature and optional token filters, samples one token, appends it to the generated sequence and repeats. The original code supports temperature, top-k and top-p sampling.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

One additional mechanism appears here: the key-value cache.

Rather than recomputing the entire preceding context for every new token, GPT-2 can retain the keys and values from previous positions, organised separately for each layer and attention head. For subsequent steps, the model computes representations for the new token while attending to the retained keys and values. The original `attn()` and `sample_sequence()` implementations explicitly support this `past`/`present` mechanism.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

For learning, I would implement generation in two stages. First, write the simplest correct version that recomputes the current context at every step. Then implement cached generation and verify that the same model and prompt produce equivalent next-token logits within numerical tolerance. That test will reveal more about positional indexing and attention than simply observing that cached generation is faster.

Caching is an optimisation, but it is important enough to the practical operation of autoregressive Transformers that I would include it in the project after the core model is correct.

## 5. Distinguish architectural requirements from engineering choices

Now that we have laid out the computation, we can begin separating what is essential to the learning objective from what serves larger-scale operation.

|
Mechanism

|

Proposed treatment in our R project

|
| --- | --- |
|

Byte-level BPE encode/decode

|

Implement and test. Use a small learned vocabulary initially.

|
|

Learned token and position embeddings

|

Implement, including the correct backward pass.

|
|

Multi-head causal attention

|

Core architectural objective. Implement and inspect every transformation.

|
|

Stacked Transformer blocks

|

Core architectural objective. Start with two blocks and configurable depth.

|
|

Pre-layer norm, residuals and final norm

|

Reproduce GPT-2’s ordering.

|
|

GELU and 4×-width feed-forward network

|

Implement and gradient-check.

|
|

Tied input/output embeddings

|

Implement and verify gradient accumulation.

|
|

Next-token loss and full backpropagation

|

Implement manually, as in previous projects.

|
|

Batched training

|

Implement, starting with small batches and explicit shapes.

|
|

Key-value caching

|

Add after verifying ordinary generation.

|
|

Original 50,257-token vocabulary and 1,024-token context

|

Historical reference; not required for the first educational run.

|
|

Original WebText corpus and parameter scale

|

Outside the initial learning objective.

|
|

GPU kernels, distributed training and mixed precision

|

Defer; these are largely computational-infrastructure concerns for this project.

|

The architectural entries are supported by the original model and tokenizer source. The proposed treatment and sequencing in the right column are design decisions for our project, not claims about how OpenAI built or trained its original models.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+2

I would also explicitly exclude modern additions from this first reproduction: rotary position embeddings, RMSNorm, grouped-query attention, mixture-of-experts layers and instruction tuning. Those may be valuable subsequent projects, but adding them now would make it harder to distinguish GPT-2’s actual mechanism from later developments.

## 6. The inspection and validation work I would build alongside it

This is where your proposed visualisation design becomes particularly valuable. With multiple heads and layers, we should not settle for an architecture diagram plus a loss curve.

I would give every major function a small, deterministic inspection example and a corresponding optional visualisation. The figures should answer concrete questions:

Tokenisation: Which characters or bytes form each token? Which ranked merges produced it? Can we recover the original text exactly?

Embeddings: What are the token, position and summed input matrices, and what happens to their dimensions?

Attention: How does the combined QKV matrix split into heads? What are the actual scores, masks, probability matrices and head outputs for a short prompt?

Layer hierarchy: How do the attention outputs merge? Where are residuals added? How do tensor shapes and selected values change from block 1 to block 2?

Learning: Do analytical and numerical gradients agree? Does the model overfit a very small training sequence? Do training and held-out validation losses behave as expected?

Generation: Which tokens were available at a prediction step, what probability did the model assign to each, which token was sampled, and how did that alter the next context?

The diagnostic code should run from a small sampled input or saved checkpoint, independently of the main training loop. Each resulting figure should be labelled as either a schematic explanation or a measurement from the implemented model.

For an educational build, I would treat these inspections and tests as deliverables in their own right. They are how we establish that the R code implements the mathematics we intend.

## 7. The build order I would use

I would resist writing the complete GPT-2 model in one pass. A more informative progression is to implement and verify the tokenizer, then a batched single-head forward pass, then multi-head attention, then a complete block with GELU and residuals, then stacked blocks and tied output weights. Only after the full forward pass is validated would I complete the manual backward pass, tiny-data learning tests, normal training, and finally cached generation.

The first major milestone should be a two-block, multi-head GPT-2-style model whose every forward transformation and analytical gradient can be inspected and tested. Once that works, longer training, larger configurations and efficiency improvements become controlled extensions of an understood system rather than additional sources of uncertainty.

The most useful immediate next step is to turn the original functions into a source-to-R implementation specification, recording each function’s mathematical purpose, tensor input/output shapes, learned parameters, forward calculation, backward calculation and minimum correctness test. That will give us a concrete architectural checklist before deciding which components deserve a full blog figure and which can remain ordinary supporting code.

