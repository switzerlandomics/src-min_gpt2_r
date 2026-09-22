# GPT-2-style decoder-only Transformer, written with base R arrays and manual
# gradients. Tensor convention: [batch, time, features]; IDs: [batch, time].
# The model is intentionally small but uses source-matched GPT-2 block ordering.

new_model_config <- function(vocab_size, context_length = 32L,
                             embedding_size = 32L, n_heads = 4L, n_layers = 2L,
                             layer_norm_epsilon = 1e-5) {
  sizes <- c(vocab_size, context_length, embedding_size, n_heads, n_layers)
  if (any(!is.finite(sizes)) || any(sizes < 1) || any(sizes != floor(sizes)) ||
      embedding_size %% n_heads != 0L || layer_norm_epsilon <= 0) {
    stop("Invalid configuration; embedding_size must be divisible by n_heads.", call. = FALSE)
  }
  list(vocab_size = as.integer(vocab_size), context_length = as.integer(context_length),
       embedding_size = as.integer(embedding_size), n_heads = as.integer(n_heads),
       n_layers = as.integer(n_layers), layer_norm_epsilon = layer_norm_epsilon)
}

initialise_model <- function(config, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  d <- config$embedding_size
  weight <- function(input_size, output_size, sd = 0.02) {
    matrix(rnorm(input_size * output_size, sd = sd), input_size, output_size)
  }
  linear <- function(input_size, output_size, sd = 0.02) {
    list(weight = weight(input_size, output_size, sd), bias = numeric(output_size))
  }
  norm <- function() list(scale = rep(1, d), bias = numeric(d))
  blocks <- lapply(seq_len(config$n_layers), function(layer) {
    list(norm1 = norm(),
         attention = list(qkv = linear(d, 3L * d),
                          projection = linear(d, d, 0.02 / sqrt(2 * config$n_layers))),
         norm2 = norm(),
         feedforward = list(up = linear(d, 4L * d),
                            down = linear(4L * d, d, 0.02 / sqrt(2 * config$n_layers))))
  })
  list(config = config,
       parameters = list(token_embedding = weight(config$vocab_size, d),
                         position_embedding = weight(config$context_length, d),
                         blocks = blocks, final_norm = norm()))
}

linear_forward <- function(input, parameters) {
  dimensions <- dim(input)
  input_matrix <- matrix(input, nrow = dimensions[1L] * dimensions[2L], ncol = dimensions[3L])
  output_matrix <- sweep(input_matrix %*% parameters$weight, 2L, parameters$bias, `+`)
  list(output = array(output_matrix, c(dimensions[1:2], ncol(output_matrix))),
       cache = list(input = input_matrix, weight = parameters$weight, dimensions = dimensions))
}

linear_backward <- function(gradient, cache) {
  n <- nrow(cache$input)
  gradient_matrix <- matrix(gradient, nrow = n)
  list(input = array(gradient_matrix %*% t(cache$weight), cache$dimensions),
       parameters = list(weight = crossprod(cache$input, gradient_matrix),
                         bias = colSums(gradient_matrix)))
}

layer_norm_forward <- function(input, parameters, epsilon = 1e-5) {
  dimensions <- dim(input)
  input_matrix <- matrix(input, nrow = prod(dimensions[1:2]), ncol = dimensions[3L])
  centred <- input_matrix - rowMeans(input_matrix)
  inverse_sd <- 1 / sqrt(rowMeans(centred^2) + epsilon)
  normalised <- centred * inverse_sd
  output <- sweep(sweep(normalised, 2L, parameters$scale, `*`),
                  2L, parameters$bias, `+`)
  list(output = array(output, dimensions),
       cache = list(normalised = normalised, inverse_sd = inverse_sd,
                    scale = parameters$scale, dimensions = dimensions))
}

layer_norm_backward <- function(gradient, cache) {
  gradient_matrix <- matrix(gradient, nrow = nrow(cache$normalised))
  scaled <- sweep(gradient_matrix, 2L, cache$scale, `*`)
  d_input <- (scaled - rowMeans(scaled) -
                cache$normalised * rowMeans(scaled * cache$normalised)) * cache$inverse_sd
  list(input = array(d_input, cache$dimensions),
       parameters = list(scale = colSums(gradient_matrix * cache$normalised),
                         bias = colSums(gradient_matrix)))
}

# The tanh approximation used in the released GPT-2 model.
gelu <- function(input) {
  0.5 * input * (1 + tanh(sqrt(2 / pi) * (input + 0.044715 * input^3)))
}

gelu_derivative <- function(input) {
  argument <- sqrt(2 / pi) * (input + 0.044715 * input^3)
  activation <- tanh(argument)
  0.5 * (1 + activation) +
    0.5 * input * (1 - activation^2) * sqrt(2 / pi) *
    (1 + 3 * 0.044715 * input^2)
}

# Q, K and V are split FIRST, then the feature dimension is separated into heads.
attention_forward <- function(input, parameters, n_heads, inspect = FALSE) {
  dimensions <- dim(input)
  b_size <- dimensions[1L]; time <- dimensions[2L]; d <- dimensions[3L]
  head_size <- d %/% n_heads
  qkv <- linear_forward(input, parameters$qkv)
  q <- qkv$output[, , seq_len(d), drop = FALSE]
  k <- qkv$output[, , d + seq_len(d), drop = FALSE]
  v <- qkv$output[, , 2L * d + seq_len(d), drop = FALSE]
  head_output <- array(0, c(b_size, time, d))
  head_cache <- vector("list", b_size * n_heads)
  snapshots <- if (inspect) vector("list", b_size * n_heads) else NULL
  for (batch in seq_len(b_size)) {
    for (head in seq_len(n_heads)) {
      indices <- seq.int((head - 1L) * head_size + 1L, length.out = head_size)
      query <- matrix(q[batch, , indices, drop = FALSE], time, head_size)
      key <- matrix(k[batch, , indices, drop = FALSE], time, head_size)
      value <- matrix(v[batch, , indices, drop = FALSE], time, head_size)
      scores <- query %*% t(key) / sqrt(head_size)
      scores[upper.tri(scores)] <- -Inf  # Mask BEFORE softmax.
      row_max <- apply(scores, 1L, max)
      exp_scores <- exp(sweep(scores, 1L, row_max, `-`))
      weights <- exp_scores / rowSums(exp_scores)
      head_output[batch, , indices] <- weights %*% value
      index <- (batch - 1L) * n_heads + head
      head_cache[[index]] <- list(query = query, key = key,
                                  value = value, weights = weights)
      if (inspect) snapshots[[index]] <- list(batch = batch, head = head,
                                             scores = scores, weights = weights)
    }
  }
  projected <- linear_forward(head_output, parameters$projection)
  list(output = projected$output,
       cache = list(qkv = qkv$cache, projection = projected$cache,
                    heads = head_cache, dimensions = dimensions, n_heads = n_heads),
       inspection = snapshots)
}

attention_backward <- function(gradient, cache) {
  projection <- linear_backward(gradient, cache$projection)
  dimensions <- cache$dimensions
  b_size <- dimensions[1L]; time <- dimensions[2L]; d <- dimensions[3L]
  head_size <- d %/% cache$n_heads
  dq <- dk <- dv <- array(0, dimensions)
  for (batch in seq_len(b_size)) {
    for (head in seq_len(cache$n_heads)) {
      indices <- seq.int((head - 1L) * head_size + 1L, length.out = head_size)
      saved <- cache$heads[[(batch - 1L) * cache$n_heads + head]]
      output_gradient <- matrix(projection$input[batch, , indices, drop = FALSE],
                                time, head_size)
      d_weights <- output_gradient %*% t(saved$value)
      d_value <- t(saved$weights) %*% output_gradient
      d_scores <- saved$weights *
        (d_weights - rowSums(d_weights * saved$weights))
      d_scores[upper.tri(d_scores)] <- 0
      dq[batch, , indices] <- d_scores %*% saved$key / sqrt(head_size)
      dk[batch, , indices] <- t(d_scores) %*% saved$query / sqrt(head_size)
      dv[batch, , indices] <- d_value
    }
  }
  d_qkv <- array(0, c(b_size, time, 3L * d))
  d_qkv[, , seq_len(d)] <- dq
  d_qkv[, , d + seq_len(d)] <- dk
  d_qkv[, , 2L * d + seq_len(d)] <- dv
  qkv <- linear_backward(d_qkv, cache$qkv)
  list(input = qkv$input,
       parameters = list(qkv = qkv$parameters, projection = projection$parameters))
}

feedforward_forward <- function(input, parameters) {
  up <- linear_forward(input, parameters$up)
  activated <- gelu(up$output)
  down <- linear_forward(activated, parameters$down)
  list(output = down$output,
       cache = list(up = up$cache, preactivation = up$output, down = down$cache))
}

feedforward_backward <- function(gradient, cache) {
  down <- linear_backward(gradient, cache$down)
  up <- linear_backward(down$input * gelu_derivative(cache$preactivation), cache$up)
  list(input = up$input, parameters = list(up = up$parameters, down = down$parameters))
}

block_forward <- function(input, parameters, config, inspect = FALSE) {
  norm1 <- layer_norm_forward(input, parameters$norm1, config$layer_norm_epsilon)
  attention <- attention_forward(norm1$output, parameters$attention, config$n_heads, inspect)
  residual1 <- input + attention$output
  norm2 <- layer_norm_forward(residual1, parameters$norm2, config$layer_norm_epsilon)
  feedforward <- feedforward_forward(norm2$output, parameters$feedforward)
  list(output = residual1 + feedforward$output,
       cache = list(norm1 = norm1$cache, attention = attention$cache,
                    norm2 = norm2$cache, feedforward = feedforward$cache),
       inspection = attention$inspection)
}

block_backward <- function(gradient, cache) {
  feedforward <- feedforward_backward(gradient, cache$feedforward)
  norm2 <- layer_norm_backward(feedforward$input, cache$norm2)
  d_residual1 <- gradient + norm2$input
  attention <- attention_backward(d_residual1, cache$attention)
  norm1 <- layer_norm_backward(attention$input, cache$norm1)
  list(input = d_residual1 + norm1$input,
       parameters = list(norm1 = norm1$parameters, attention = attention$parameters,
                         norm2 = norm2$parameters, feedforward = feedforward$parameters))
}

model_forward <- function(model, token_ids, inspect = FALSE) {
  config <- model$config; parameters <- model$parameters
  if (!is.matrix(token_ids) || !is.numeric(token_ids) || anyNA(token_ids) ||
      any(token_ids != floor(token_ids)) ||
      any(token_ids < 1L | token_ids > config$vocab_size)) {
    stop("token_ids must be a matrix of valid one-based token IDs.", call. = FALSE)
  }
  b_size <- nrow(token_ids); time <- ncol(token_ids); d <- config$embedding_size
  if (b_size < 1L || time < 1L || time > config$context_length) {
    stop("Invalid batch size or context length.", call. = FALSE)
  }
  hidden <- array(0, c(b_size, time, d))
  for (batch in seq_len(b_size)) {
    for (position in seq_len(time)) {
      hidden[batch, position, ] <-
        parameters$token_embedding[token_ids[batch, position], ] +
        parameters$position_embedding[position, ]
    }
  }
  block_cache <- vector("list", config$n_layers)
  snapshots <- if (inspect) vector("list", config$n_layers) else NULL
  for (layer in seq_len(config$n_layers)) {
    result <- block_forward(hidden, parameters$blocks[[layer]], config, inspect)
    hidden <- result$output
    block_cache[[layer]] <- result$cache
    if (inspect) snapshots[[layer]] <- list(attention = result$inspection,
                                             hidden_states = hidden)
  }
  final_norm <- layer_norm_forward(hidden, parameters$final_norm, config$layer_norm_epsilon)
  features <- matrix(final_norm$output, nrow = b_size * time, ncol = d)
  logits <- array(features %*% t(parameters$token_embedding),
                  c(b_size, time, config$vocab_size))
  list(logits = logits,
       cache = list(token_ids = token_ids, blocks = block_cache, final_norm = final_norm$cache,
                    features = features),
       inspection = snapshots)
}

cross_entropy <- function(logits, targets, gradient = FALSE) {
  dimensions <- dim(logits)
  if (length(dimensions) != 3L || !is.matrix(targets) ||
      !identical(dim(targets), dimensions[1:2]) || anyNA(targets) ||
      any(targets != floor(targets)) ||
      any(targets < 1L | targets > dimensions[3L])) {
    stop("targets must be a valid [batch, time] ID matrix.", call. = FALSE)
  }
  n <- prod(dimensions[1:2]); vocab_size <- dimensions[3L]
  scores <- matrix(logits, nrow = n, ncol = vocab_size)
  row_max <- apply(scores, 1L, max)
  shifted <- sweep(scores, 1L, row_max, `-`)
  normaliser <- log(rowSums(exp(shifted)))
  selected <- cbind(seq_len(n), as.vector(targets))
  loss <- mean(normaliser - shifted[selected])
  if (!gradient) return(loss)
  probabilities <- exp(shifted - normaliser)
  probabilities[selected] <- probabilities[selected] - 1
  list(loss = loss, gradient = array(probabilities / n, dimensions))
}

model_backward <- function(model, forward, targets) {
  parameters <- model$parameters
  objective <- cross_entropy(forward$logits, targets, gradient = TRUE)
  d_logits <- matrix(objective$gradient, nrow = nrow(forward$cache$features))
  d_embeddings_output <- t(d_logits) %*% forward$cache$features
  d_features <- d_logits %*% parameters$token_embedding
  d_hidden <- array(d_features, forward$cache$final_norm$dimensions)
  final_norm <- layer_norm_backward(d_hidden, forward$cache$final_norm)
  d_hidden <- final_norm$input
  block_gradients <- vector("list", model$config$n_layers)
  for (layer in rev(seq_len(model$config$n_layers))) {
    result <- block_backward(d_hidden, forward$cache$blocks[[layer]])
    d_hidden <- result$input
    block_gradients[[layer]] <- result$parameters
  }
  token_gradients <- d_embeddings_output  # Tied embedding output projection.
  position_gradients <- parameters$position_embedding * 0
  ids <- forward$cache$token_ids
  for (batch in seq_len(nrow(ids))) {
    for (position in seq_len(ncol(ids))) {
      token_gradients[ids[batch, position], ] <-
        token_gradients[ids[batch, position], ] + d_hidden[batch, position, ]
      position_gradients[position, ] <-
        position_gradients[position, ] + d_hidden[batch, position, ]
    }
  }
  list(loss = objective$loss,
       gradients = list(token_embedding = token_gradients,
                        position_embedding = position_gradients,
                        blocks = block_gradients, final_norm = final_norm$parameters))
}

model_loss <- function(model, token_ids, targets) {
  cross_entropy(model_forward(model, token_ids)$logits, targets)
}

parameter_count <- function(model) length(unlist(model$parameters, use.names = FALSE))
