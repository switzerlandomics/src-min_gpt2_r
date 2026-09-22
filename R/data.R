# Corpus and batching. Files are read as raw bytes so no text normalisation occurs.
read_corpus <- function(path, tokenizer) {
  if (!file.exists(path)) stop("Corpus file not found: ", path, call. = FALSE)
  size <- file.info(path)$size
  if (!is.finite(size) || size < 1) stop("Corpus must be nonempty.", call. = FALSE)
  connection <- file(path, open = "rb")
  on.exit(close(connection))
  encode_bytes(readBin(connection, what = "raw", n = size), tokenizer)
}

split_corpus <- function(token_ids, fractions = c(train = 0.8, validation = 0.1, test = 0.1)) {
  if (length(token_ids) < 30L) stop("Corpus is too short for three splits.", call. = FALSE)
  if (length(fractions) != 3L || any(!is.finite(fractions)) ||
      any(fractions <= 0) || abs(sum(fractions) - 1) > 1e-8) {
    stop("fractions must be three positive values summing to one.", call. = FALSE)
  }
  n <- length(token_ids)
  train_end <- floor(n * fractions[1L])
  validation_end <- train_end + floor(n * fractions[2L])
  list(train = token_ids[seq_len(train_end)],
       validation = token_ids[seq.int(train_end + 1L, validation_end)],
       test = token_ids[seq.int(validation_end + 1L, n)])
}

make_batch <- function(token_ids, context_length, batch_size = 1L, starts = NULL) {
  n <- length(token_ids)
  if (length(context_length) != 1L || context_length < 1L ||
      context_length != floor(context_length) || n <= context_length) {
    stop("context_length must be a positive integer smaller than the split.", call. = FALSE)
  }
  if (is.null(starts)) {
    if (batch_size < 1L || batch_size != floor(batch_size)) stop("Invalid batch_size.", call. = FALSE)
    starts <- sample.int(n - context_length, as.integer(batch_size), replace = TRUE)
  }
  starts <- as.integer(starts)
  if (!length(starts) || anyNA(starts) ||
      any(starts < 1L | starts > n - context_length)) {
    stop("Invalid batch start position(s).", call. = FALSE)
  }
  batch_size <- length(starts)
  inputs <- targets <- matrix(0L, nrow = batch_size, ncol = context_length)
  for (row in seq_len(batch_size)) {
    positions <- seq.int(starts[row], length.out = context_length)
    inputs[row, ] <- token_ids[positions]
    targets[row, ] <- token_ids[positions + 1L]
  }
  list(token_ids = inputs, targets = targets, starts = starts)
}

validation_starts <- function(token_ids, context_length, max_windows = 8L) {
  if (length(token_ids) <= context_length) stop("Evaluation split is too short.", call. = FALSE)
  possible <- seq.int(1L, length(token_ids) - context_length, by = context_length)
  indices <- unique(as.integer(round(seq(1, length(possible), length.out =
                                          min(length(possible), as.integer(max_windows))))))
  possible[indices]
}
