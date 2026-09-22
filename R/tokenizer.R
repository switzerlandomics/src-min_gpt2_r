# Byte-level tokenizer. Token IDs are one-based: raw byte 0 = ID 1.
# Byte-level BPE can later implement the same encode_text/decode_tokens API.
new_byte_tokenizer <- function() {
  structure(list(type = "byte", vocab_size = 256L), class = "sequence_tokenizer")
}

validate_tokenizer <- function(tokenizer) {
  if (!inherits(tokenizer, "sequence_tokenizer") ||
      !identical(tokenizer$type, "byte") ||
      !identical(tokenizer$vocab_size, 256L)) {
    stop("Expected a tokenizer created by new_byte_tokenizer().", call. = FALSE)
  }
  invisible(TRUE)
}

vocabulary_size <- function(tokenizer) {
  validate_tokenizer(tokenizer)
  tokenizer$vocab_size
}

encode_bytes <- function(bytes, tokenizer) {
  validate_tokenizer(tokenizer)
  if (!is.raw(bytes)) stop("bytes must be a raw vector.", call. = FALSE)
  as.integer(bytes) + 1L
}

decode_bytes <- function(token_ids, tokenizer) {
  validate_tokenizer(tokenizer)
  if (!is.numeric(token_ids) || anyNA(token_ids) ||
      any(!is.finite(token_ids)) || any(token_ids != floor(token_ids)) ||
      any(token_ids < 1L | token_ids > tokenizer$vocab_size)) {
    stop("token_ids must be whole-number IDs from 1 to 256.", call. = FALSE)
  }
  as.raw(as.integer(token_ids) - 1L)
}

encode_text <- function(text, tokenizer) {
  if (!is.character(text) || length(text) != 1L || is.na(text)) {
    stop("text must be one non-missing character string.", call. = FALSE)
  }
  encode_bytes(charToRaw(enc2utf8(text)), tokenizer)
}

decode_tokens <- function(token_ids, tokenizer, incomplete = c("error", "replace")) {
  incomplete <- match.arg(incomplete)
  bytes <- decode_bytes(token_ids, tokenizer)
  if (!length(bytes)) return("")
  # NUL cannot be represented safely in an ordinary R character scalar.
  if (any(bytes == as.raw(0L))) {
    if (incomplete == "error") stop("NUL byte in token sequence; use decode_bytes().", call. = FALSE)
    bytes[bytes == as.raw(0L)] <- as.raw(63L)
  }
  text <- rawToChar(bytes)
  decoded <- iconv(text, from = "UTF-8", to = "UTF-8",
                   sub = if (incomplete == "replace") "byte" else NA_character_)
  if (is.na(decoded)) {
    stop("Incomplete or invalid UTF-8; use decode_bytes() or incomplete='replace'.",
         call. = FALSE)
  }
  decoded
}
