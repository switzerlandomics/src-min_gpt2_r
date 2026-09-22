# GPT-2 in R: findings from the original sources and a preliminary project plan

I have worked through the original GPT-2 model, tokenizer and sampling source, alongside the 2019 paper, OpenAI’s three release articles and the Hugging Face model card. These sources give us a sufficiently concrete foundation to define the next R project without relying on simplified accounts of how GPT-2 works.

The central finding is that GPT-2 is a manageable architectural step beyond your first Transformer, but a substantially larger implementation task. The prediction objective is unchanged. The important additions are byte-level tokenisation, multi-head attention, multiple Transformer blocks, the precise GPT-2 layer operations, and their gradients. Much of the remaining complexity concerns making the resulting model train efficiently at scale.

There is also an important source distinction: OpenAI released the model’s forward computation, tokenizer, generation code and trained checkpoints, but not a complete implementation of its original pretraining pipeline. We can reproduce the published architecture accurately while being explicit that our dataset preparation, training loop and some optimisation choices are our own.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+4

## 1. What the original project actually established

The paper’s question was broader than whether a Transformer could generate plausible text. OpenAI investigated whether a model trained on one general objective, predicting the next token, could acquire useful behaviours for other language tasks without being trained separately for each task.

The paper reports experiments with four model sizes trained on WebText, a corpus containing slightly more than eight million documents and approximately 40 GB of text after filtering. It evaluates language modelling and several other tasks, including question answering, translation and summarisation, without task-specific fine-tuning. Those experimental results concern models trained at substantial scale, not capabilities that automatically follow from implementing the same architecture.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+2

For our project, this suggests a useful separation between understanding the mechanism and investigating what larger-scale training produces. We should first establish the former using a model small enough to inspect completely.

### Historical context from the release articles

The three OpenAI articles help explain why the repository and model releases developed as they did.

OpenAI introduced GPT-2 publicly in February 2019, initially releasing a smaller model rather than its largest trained checkpoint. The August follow-up discussed its staged-release process and announced the 774M model. In November, OpenAI released the largest, approximately 1.5-billion-parameter version. These articles primarily document model capabilities, release decisions and related research; they are not substitutes for the source code when specifying individual numerical operations.

![](https://www.google.com/s2/favicons?domain=https://openai.com\&sz=32)

OpenAI

+2

The repository also corrects a historical naming issue: the paper’s original parameter counts were inaccurate. The smallest model, commonly described in the original paper as 117M, is labelled 124M in the released repository and Hugging Face model card. We should use 124M when identifying that released checkpoint, while preserving the paper’s original terminology when discussing its historical results.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+2

## 2. The original source files and what each contributes

The most useful implementation reference is [`src/model.py`](https://github.com/openai/gpt-2/blob/master/src/model.py) . It is remarkably compact: the actual forward computation is defined through a small collection of functions, rather than a large library of abstract model classes.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

|
Original source

|

What we should learn from it

|
| --- | --- |
|

[`model.py`](https://github.com/openai/gpt-2/blob/master/src/model.py) 

|

Parameter shapes, embeddings, normalisation, attention, feed-forward network, block stacking, output logits and cached keys and values.

|
|

[`encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py) 

|

UTF-8 byte representation, text splitting, ranked BPE merges, token IDs and reversible decoding.

|
|

[`sample.py`](https://github.com/openai/gpt-2/blob/master/src/sample.py) 

|

Autoregressive generation, temperature, top-k and top-p filtering, and use of the key-value cache.

|
|

[`interactive_conditional_samples.py`](https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py) 

|

How the tokenizer, model configuration, saved parameters, input prompt and sampling routine are assembled into a working application.

|

These four files give us a direct route from the individual mathematical functions to a complete text-generation programme.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

+3

The original source uses TensorFlow 1 and represents many learned linear transformations through a function named `conv1d()`. In this implementation, that function performs a learned linear projection at each position; it is not a convolution across neighbouring tokens. Its weight tensor has a leading dimension of one and is reshaped for matrix multiplication. In R, an ordinary matrix multiplication with an appropriate weight matrix and bias would express the same operation more clearly.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

That is an excellent example of how our code can differ in presentation from the original while preserving the mathematics exactly.

## 3. The precise model we should reproduce

The released model’s computation can be summarised as follows:

Input

Token IDs → learned token embeddings + learned positional embeddings

Transformer block, repeated with separate learned parameters

Layer norm → multi-head causal attention → residual addition

Layer norm → 4×-width feed-forward network with GELU → residual addition

Output

Final layer norm → projection using tied token-embedding weights → vocabulary logits

That ordering is not a generic Transformer sketch: it follows the original `model()`, `block()`, `attn()` and `mlp()` functions.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

A few source-level details deserve particular attention because they would be easy to omit in an otherwise plausible reproduction.

Multi-head attention. GPT-2 calculates queries, keys and values using one combined projection, then splits the result into separate heads. Each head performs causal scaled dot-product attention. The outputs are merged and passed through another learned projection. The scaling factor uses the square root of the dimension per head, not the full model width.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

Normalisation and residuals. GPT-2 layer-normalises before each attention and feed-forward sublayer, adds the sublayer result to the unnormalised residual stream, and applies an additional final layer norm. The original source uses an epsilon of `1e-5` for layer normalisation.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

Feed-forward network. Each block expands its representation to four times the model width, applies GELU, and projects it back to the original width. The original code uses a particular tanh-based GELU approximation. If we want to reproduce that implementation, we should use that formula and differentiate it correctly, rather than substituting ReLU or a different GELU implementation without noting the change.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

Output projection. GPT-2 reuses the token-embedding matrix to calculate vocabulary logits. It does not learn a separate output-weight matrix in `model.py`. For our hand-written backward pass, this means gradients arising from the input embeddings and output projection must both contribute to the same parameter matrix.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

Initialisation. The paper describes scaling residual-layer weights to account for increasing depth, while the released `model.py` specifies the initialisers used when constructing its inference graph. We should record both sources and avoid assuming that the released inference file fully documents the original training-time initialisation procedure.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+1

Your existing single-block Transformer already implements several of these ideas. The challenge is not to start again, but to preserve the correct operations while extending them to multiple heads, multiple blocks and a token-based vocabulary.

## 4. Tokenisation needs its own investigation

I would treat the tokenizer as a substantial, independent part of the next project, rather than a short preprocessing function.

The paper explains why GPT-2 uses byte-level BPE: it combines the ability to represent arbitrary text with tokens that can correspond to frequently occurring multi-character sequences. The original `encoder.py` makes that idea concrete through a reversible byte-to-Unicode mapping, a text-splitting regular expression, ranked BPE merges and a token-to-ID vocabulary.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+1

One subtle but important distinction emerged from the source review: the released encoder applies existing BPE merge rules; it does not train those rules from a raw text corpus. If we want a completely first-principles tokenizer, we should implement its vocabulary-training process as an additional R component and test the resulting encoder and decoder together. We should distinguish that educational implementation from an exact reproduction of OpenAI’s original trained vocabulary.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

The simplest useful experiment would follow a short text through its UTF-8 bytes, initial symbols, successive merges and final token IDs. Unlike a schematic tokenizer diagram, this can produce a figure derived directly from a recorded transformation.

I would also keep the published GPT-2 tokenizer compatibility test separate from our small educational vocabulary. A smaller vocabulary makes training affordable, but token IDs, token boundaries and model outputs will differ from those of the original checkpoint.

## 5. What Hugging Face adds to the project

The [Hugging Face GPT-2 release](https://huggingface.co/openai-community/gpt2)  provides a practical, independently accessible reference for the released 124M pretrained model. Its model card documents the causal language-modelling objective, 50,257-token vocabulary, 1,024-token context and available pretrained weights. It also explains that the original WebText training corpus was not publicly released.

![](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

This suggests a valuable optional verification exercise after our R implementation is complete.

We could construct a configuration matching the published small model, load its tokenizer and pretrained parameter values, and compare the R forward-pass logits against those from an established reference implementation for the same token IDs. That would test much more than whether our small R model can learn a simple text corpus: it would test whether we have reproduced GPT-2’s numerical computation closely enough to use its actual released parameters.

This would require careful conversion of weight names, array dimensions, tensor layouts and numerical conventions. It is not necessary for the first training milestone, and loading published weights for a compatibility test should not be confused with using pretrained weights to train our own model from scratch.

There is a practical distinction here: the Hugging Face model card and documentation were accessible in this review, but I have not yet inspected or numerically compared the individual checkpoint tensors. We should not claim weight-level equivalence until we have performed that test.

![](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

+1

## 6. What the sources do not establish

The paper documents the model’s research objective, data preparation at a high level, architectural changes and experimental results. The released repository documents inference and generation. Neither supplies a complete, executable reproduction of the original WebText pretraining run, including every original data-processing decision, optimiser setting, learning-rate schedule and checkpointing procedure. The model card also notes that exact training details were not fully disclosed.

![](https://www.google.com/s2/favicons?domain=https://cdn.openai.com\&sz=32)

cdn.openai.com

+2

Consequently, our design document should explicitly distinguish source-matched architecture from project-defined training infrastructure.

The learning objective is clear: predict the following token from the preceding context. We can implement its loss, gradients and optimiser ourselves, preserve the validation discipline from your previous projects, and document our choices without presenting them as an exact reconstruction of OpenAI’s original pretraining run.

## 7. Proposed direction for the R project

My proposed scope is now more precise: build a small, configurable GPT-2-style model whose architecture follows OpenAI’s original `model.py`, while implementing the complete learning algorithm and diagnostic tools ourselves in R.

We should use standard R matrices and arrays for numerical operations, alongside familiar packages such as `data.table`, `dplyr` and `ggplot2` where they improve data handling, experimental analysis and plotting. No packaged Transformer, automatic differentiation or pretrained weights are needed for the main educational training path.

The build should proceed through three distinct milestones.

First, establish architectural correctness. Implement and test the tokenizer; extend the current model to batched multi-head attention; reproduce GPT-2’s block ordering, GELU, final normalisation and tied embeddings; and stack multiple blocks. Inspect each transformation using a tiny deterministic input.

Second, establish learning correctness. Complete the manual backward pass, verify gradients numerically, demonstrate that the model can overfit a tiny token sequence, then run a properly separated training and validation experiment. Keep diagnostic capture optional so it does not slow ordinary training.

Third, establish inference compatibility. Implement text generation first by recomputing the full context, then add GPT-2’s key-value caching and compare the resulting logits. If useful, extend the tests to the released 124M checkpoint, preserving a clear separation between our trained model and the published reference model. The original `sample.py` provides the specification for cached generation and sampling options.

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw.githubusercontent.com

+1

The crucial design decision I would make before coding is to define two different meanings of “from scratch” in the project documentation: the educational model learns its own parameters from data with an explicitly implemented training algorithm; the reference compatibility test, if undertaken, may load published GPT-2 parameters solely to verify that our R calculations match the original architecture.

That gives us a rigorous project with a clear destination. We are not simply making the previous Transformer larger. We are learning how the individual attention calculation becomes a multi-head module, how modules become stacked blocks, how the complete network learns from tokens, and how its computation can be checked against a real, released GPT-2 model.

