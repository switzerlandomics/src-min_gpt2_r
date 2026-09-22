# Run from repository root:
# Rscript tests/test_model.R

source("R/tokenizer.R")
source("R/data.R")
source("R/model.R")


model <- initialise_model(
  new_model_config(
    vocab_size = 256L,
    context_length = 8L,
    embedding_size = 8L,
    n_heads = 2L,
    n_layers = 2L
  ),
  seed = 11L
)

inputs <- matrix(
  c(65L, 66L, 67L, 68L, 69L, 70L),
  nrow = 2L
)

targets <- inputs + 1L


# ---------------------------------------------------------------------------
# Forward pass and inspection
# ---------------------------------------------------------------------------

result <- model_forward(
  model,
  inputs,
  inspect = TRUE
)

stopifnot(
  identical(
    dim(result$logits),
    c(2L, 3L, 256L)
  )
)

stopifnot(
  length(result$inspection) == 2L
)

for (layer in result$inspection) {

  stopifnot(
    length(layer$attention) == 4L
  )

  stopifnot(
    identical(
      dim(layer$hidden_states),
      c(2L, 3L, 8L)
    )
  )

  for (item in layer$attention) {

    weights <- item$weights
    scores <- item$scores

    stopifnot(
      identical(dim(weights), c(3L, 3L)),
      identical(dim(scores), c(3L, 3L))
    )

    stopifnot(
      max(abs(rowSums(weights) - 1)) < 1e-12
    )

    stopifnot(
      all(weights[upper.tri(weights)] == 0)
    )

    stopifnot(
      all(is.infinite(scores[upper.tri(scores)]))
    )

    stopifnot(
      all(is.finite(weights))
    )
  }
}


# ---------------------------------------------------------------------------
# Loss and backward pass
# ---------------------------------------------------------------------------

loss <- cross_entropy(
  result$logits,
  targets
)

stopifnot(
  is.finite(loss),
  loss > 0
)

backward <- model_backward(
  model,
  result,
  targets
)

stopifnot(
  is.finite(backward$loss)
)

stopifnot(
  isTRUE(
    all.equal(
      backward$loss,
      loss,
      tolerance = 1e-12
    )
  )
)

stopifnot(
  identical(
    dim(backward$gradients$token_embedding),
    c(256L, 8L)
  )
)

stopifnot(
  identical(
    dim(
      backward$gradients$blocks[[1L]]$attention$qkv$weight
    ),
    c(8L, 24L)
  )
)


# ---------------------------------------------------------------------------
# Causal independence
# ---------------------------------------------------------------------------

# Changing future tokens must not alter logits at earlier positions.

changed <- inputs
changed[, 3L] <- changed[, 3L] + 10L

alternative <- model_forward(
  model,
  changed
)

stopifnot(
  max(
    abs(
      result$logits[, 1:2, ] -
        alternative$logits[, 1:2, ]
    )
  ) < 1e-12
)


# ---------------------------------------------------------------------------
# Configurable attention heads
# ---------------------------------------------------------------------------

model_one_head <- initialise_model(
  new_model_config(
    vocab_size = 256L,
    context_length = 8L,
    embedding_size = 8L,
    n_heads = 1L,
    n_layers = 1L
  ),
  seed = 12L
)

stopifnot(
  identical(
    dim(
      model_forward(
        model_one_head,
        inputs
      )$logits
    ),
    c(2L, 3L, 256L)
  )
)


# ---------------------------------------------------------------------------
# Plotting data
# ---------------------------------------------------------------------------

# Plotting functions are optional and must not alter model computations.

source("R/plots.R")

tokenizer <- new_byte_tokenizer()
prompt_ids <- as.integer(inputs[1L, ])

inspection_data <- attention_plot_data(
  model,
  prompt_ids,
  layer = 1L
)

stopifnot(
  nrow(inspection_data) ==
    model$config$n_heads * length(prompt_ids)^2
)

stopifnot(
  length(unique(inspection_data$head)) ==
    model$config$n_heads
)

# Masked future cells are NA in plotting data (not observed weight zero).
observed <- !is.na(inspection_data$weight)
stopifnot(
  any(!observed),
  all(inspection_data$weight[observed] >= 0 &
      inspection_data$weight[observed] <= 1)
)

probabilities <- next_token_probabilities(
  model,
  prompt_ids
)

stopifnot(
  length(probabilities) == 256L,
  all(is.finite(probabilities)),
  all(probabilities >= 0),
  abs(sum(probabilities) - 1) < 1e-12
)


cat(
  "PASS: model shapes, gradients, causal masking and inspection data\n"
)
