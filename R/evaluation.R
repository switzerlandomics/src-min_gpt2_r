# Evaluation and straightforward autoregressive sampling (no KV cache yet).
evaluate_model <- function(model, token_ids, context_length = model$config$context_length,
                           max_windows = 8L, batch_size = 2L) {
  starts <- validation_starts(token_ids, context_length, max_windows)
  groups <- split(starts, ceiling(seq_along(starts) / batch_size))
  total_loss <- 0
  total_tokens <- 0L
  for (group in groups) {
    batch <- make_batch(token_ids, context_length, starts = group)
    loss <- model_loss(model, batch$token_ids, batch$targets)
    count <- length(batch$targets)
    total_loss <- total_loss + loss * count
    total_tokens <- total_tokens + count
  }
  total_loss / total_tokens
}

# Keep sampling RNG separate from the RNG used to sample training batches.
with_local_seed <- function(seed, expression) {
  if (is.null(seed)) return(force(expression))
  existed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (existed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (existed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  })
  set.seed(seed)
  force(expression)
}

sample_token <- function(logits, temperature = 1, top_k = NULL) {
  if (length(temperature) != 1L || !is.finite(temperature) || temperature <= 0) {
    stop("temperature must be positive.", call. = FALSE)
  }
  scores <- as.numeric(logits) / temperature
  if (!is.null(top_k)) {
    if (length(top_k) != 1L || top_k < 1 || top_k != floor(top_k)) stop("Invalid top_k.")
    keep <- order(scores, decreasing = TRUE)[seq_len(min(length(scores), top_k))]
    scores[-keep] <- -Inf
  }
  shifted <- scores - max(scores)
  probabilities <- exp(shifted)
  probabilities <- probabilities / sum(probabilities)
  sample.int(length(probabilities), 1L, prob = probabilities)
}

generate_tokens <- function(model, prompt_ids, n_tokens = 80L,
                            temperature = 1, top_k = NULL, seed = NULL) {
  if (!length(prompt_ids) || anyNA(prompt_ids) ||
      any(prompt_ids < 1L | prompt_ids > model$config$vocab_size)) {
    stop("prompt_ids must contain at least one valid token ID.", call. = FALSE)
  }
  if (n_tokens < 0L || n_tokens != floor(n_tokens)) stop("Invalid n_tokens.")
  with_local_seed(seed, {
    generated <- as.integer(prompt_ids)
    if (n_tokens > 0L) for (step in seq_len(n_tokens)) {
      context <- tail(generated, model$config$context_length)
      inputs <- matrix(context, nrow = 1L)
      forward <- model_forward(model, inputs)
      next_logits <- forward$logits[1L, length(context), ]
      generated <- c(generated, sample_token(next_logits, temperature, top_k))
    }
    generated
  })
}
