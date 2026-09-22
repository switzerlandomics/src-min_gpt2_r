# GPT-2-style language model in R: preliminary design

**Status:** Initial design proposal. The architecture, scope, dependencies and experiment settings may change as we implement and test the model.

## Purpose

This project follows our character-level RNN and single-head, single-block Transformer. The aim is to understand how the familiar next-token prediction task develops into a deeper, GPT-2-style decoder-only language model. We will implement the essential mechanisms ourselves, verify them against the mathematics and make their intermediate computations inspectable. The objective is a correct, understandable and reproducible small model, not a reproduction of GPT-2's original scale, pretrained weights or language quality. We will continue to use ordinary text so that the algorithm can be understood before introducing the additional assumptions required by biological data.

## Proposed model

The main additions are byte-level byte-pair encoding (BPE) tokenisation, multiple causal attention heads, stacked Transformer blocks and a GPT-2-style position-wise feed-forward network using GELU. Each head will calculate scaled dot-product attention over its permitted context; the heads will be concatenated and projected before passing through residual connections and subsequent blocks. We plan to use learned token and positional embeddings, layer normalisation, a final vocabulary projection and autoregressive next-token sampling. The code should reflect this hierarchy clearly: individual operations form attention heads, heads form attention modules, attention and feed-forward modules form blocks, and blocks form the complete model. We will document the precise equations, tensor shapes and differences between our implementation and the published GPT-2 architecture.

We will begin with a small, configurable model, initially using a short context, two blocks and multiple heads. Tokenisation and each new module will be implemented and tested independently before the full training run. Configuration and corpus choices will be determined by those tests rather than fixed prematurely. Increasing parameter count or context length is a later experiment, not a substitute for checking that the small model is correct.

## Implementation and dependencies

*From scratch* applies to the learning algorithm, not to routine data handling. We may use familiar R packages such as `dplyr`, `data.table`, `ggplot2`, `readr` and `svglite` where they make supporting code clearer, particularly for preparing data, recording experiments and making figures. The numerical model should continue to use readable R matrix and array operations. We will implement tokenisation, embeddings, attention, normalisation, feed-forward transformations, loss, backward derivatives, optimisation and generation ourselves. We will not delegate these calculations to a deep-learning framework, automatic differentiation, pretrained model or packaged Transformer layer.

The main training path should remain compact and efficient. Optional inspection code will be kept separate so that small, representative inputs or saved checkpoints can be used to display tokenisation, intermediate matrices, attention masks and weights, head concatenation, residual additions, representations across blocks, and changes in next-token probabilities. Diagnostic figures must distinguish schematic explanations from actual model values and should not be generated at every training update.

## Verification and experiments

We will start with deterministic tests of tokenisation, tensor dimensions, causal masking, forward computations and generation. Analytical gradients should be checked against numerical derivatives on tiny models, including across multiple heads, residual paths and stacked blocks. We will retain explicit train, validation and test separation, checkpoint selection using held-out validation, recorded training and generation settings, and resumable experiment state. Each run should save both the original command and a reconstructable command with resolved defaults and any changes made on resumption. Learning curves, baseline comparisons and controlled generation examples will come from recorded outputs, not illustrative numbers.

## What comes next

Once the architecture is correct and its behaviour measured, we can decide whether to explore larger configurations or move on to more recent Transformer mechanisms and model adaptation. Later biological applications will require their own representations, objectives, evaluation and evidence standards; a successful text-generation demonstration does not by itself establish suitability for biological or clinical use.
