#!/usr/bin/env Rscript
# Show how the project's byte tokenizer converts text and real corpus data into
# the integer input/target matrices used for next-token prediction.
# Run from the repository root. The report is Markdown suitable for a blog.

source("R/tokenizer.R")
source("R/data.R")

options <- list(input = "data/tiny_shakespeare.txt", text = "you are,",
                context_length = 8L, start = 1L, output = "")

for (argument in commandArgs(trailingOnly = TRUE)) {
  if (argument == "--help") {
    cat("Usage: Rscript experiments/inspect_tokenizer.R \\\n",
        "  [--input=FILE] [--text=TEXT] [--context-length=N] \\\n",
        "  [--start=N] [--output=FILE]\n\n",
        "Run from the repository root. The report is printed as Markdown.\n",
        "--output also saves the same report to a Markdown file.\n", sep = "")
    quit(save = "no", status = 0L)
  }
  parts <- regmatches(argument, regexec("^--([a-z][a-z0-9-]*)=(.*)$", argument))[[1L]]
  if (length(parts) != 3L) stop("Unknown argument: ", argument, call. = FALSE)
  key <- gsub("-", "_", parts[2L])
  if (!key %in% names(options)) stop("Unknown option: ", key, call. = FALSE)
  value <- parts[3L]
  if (key %in% c("context_length", "start")) {
    value <- suppressWarnings(as.integer(value))
    if (is.na(value) || value < 1L) {
      stop(key, " must be a positive integer.", call. = FALSE)
    }
  }
  options[[key]] <- value
}

if (!nzchar(options$text)) stop("--text must be nonempty.", call. = FALSE)
if (!nzchar(options$input) || !file.exists(options$input)) {
  stop("Corpus not found: ", options$input,
       ". Supply a real file with --input=FILE (or place Tiny Shakespeare in data/).",
       call. = FALSE)
}

tokenizer <- new_byte_tokenizer()
example_ids <- encode_text(options$text, tokenizer)
stopifnot(identical(enc2utf8(decode_tokens(example_ids, tokenizer)),
                    enc2utf8(options$text)))
corpus_ids <- read_corpus(options$input, tokenizer)
splits <- split_corpus(corpus_ids)
if (length(splits$train) <= options$context_length ||
    options$start > length(splits$train) - options$context_length) {
  stop("The requested training window exceeds the training split.", call. = FALSE)
}
batch <- make_batch(splits$train, context_length = options$context_length,
                    starts = options$start)

# These labels describe *bytes*, which are not necessarily full characters.
byte_label <- function(token_id) {
  value <- as.integer(decode_bytes(token_id, tokenizer))
  if (value == 32L) return("SPACE")
  if (value == 10L) return("\\n")
  if (value == 13L) return("\\r")
  if (value == 9L) return("\\t")
  if (value == 124L) return("PIPE")    # Avoid an unescaped Markdown table delimiter.
  if (value == 96L) return("BACKTICK")
  if (value == 92L) return("BACKSLASH")
  if (value == 60L) return("&lt;")
  if (value == 62L) return("&gt;")
  if (value == 38L) return("&amp;")
  if (value >= 33L && value <= 126L) return(intToUtf8(value))
  sprintf("0x%02X", value)
}

hex_byte <- function(token_id) sprintf("%02X", as.integer(decode_bytes(token_id, tokenizer)))
printable_text <- function(ids) {
  encodeString(decode_tokens(ids, tokenizer, incomplete = "replace"), quote = '"')
}

lines <- character()
emit <- function(...) lines <<- c(lines, ...)

emit("# From text to next-token training data", "",
     "This report uses the project's **actual byte tokenizer and batching functions**.",
     "Each token represents one raw byte; token ID = byte value + 1 for R's one-based indexing.",
     "This is **not yet GPT-2's byte-pair encoding (BPE)**.", "")

emit("## 1. Encode a readable input", "",
     paste0("Input text: ", encodeString(options$text, quote = '"'), "  "),
     paste0("Vocabulary: ", vocabulary_size(tokenizer), " possible byte tokens.  "),
     paste0("UTF-8 bytes (hex): `", paste(vapply(example_ids, hex_byte, character(1L)), collapse = " "), "`  "),
     paste0("Token IDs (R): `", paste(example_ids, collapse = " "), "`  "),
     paste0("Decoded round-trip: ", printable_text(example_ids)), "",
     "| Position | Byte shown | Hex byte | Token ID |", 
     "|---:|:---|:---:|---:|")
for (position in seq_along(example_ids)) {
  id <- example_ids[position]
  emit(sprintf("| %d | %s | %s | %d |", position, byte_label(id), hex_byte(id), id))
}

emit("", "## 2. Read the real corpus", "",
     paste0("Corpus file: `", basename(options$input), "`  "),
     paste0("Total tokens (bytes): ", format(length(corpus_ids), big.mark = ","), "  "),
     paste0("Contiguous split sizes — train: ", format(length(splits$train), big.mark = ","),
            "; validation: ", format(length(splits$validation), big.mark = ","),
            "; test: ", format(length(splits$test), big.mark = ","), "."), "",
     "`read_corpus()` reads the original file as raw bytes without normalisation.",
     "`split_corpus()` holds out the later 10% for validation and the final 10% for testing.", "")

validation_prompt <- head(splits$validation, 8L)
emit("The first eight bytes of this corpus's validation split (the experiment's inspection prompt):  ",
     paste0("Decoded: ", printable_text(validation_prompt), "  "),
     paste0("Token IDs: `", paste(validation_prompt, collapse = " "), "`"), "")

emit("## 3. Create one real training window", "",
     paste0("Training split, starting byte position: ", options$start, "  "),
     paste0("Context length: ", options$context_length, "  "),
     paste0("Model input: ", printable_text(as.integer(batch$token_ids[1L, ])), "  "),
     paste0("Next-token targets: ", printable_text(as.integer(batch$targets[1L, ]))), "",
     "| Position | Input byte | Input hex | Input ID | Next byte (target) | Target hex | Target ID |",
     "|---:|:---|:---:|---:|:---|:---:|---:|")
for (position in seq_len(options$context_length)) {
  input_id <- batch$token_ids[1L, position]
  target_id <- batch$targets[1L, position]
  emit(sprintf("| %d | %s | %s | %d | %s | %s | %d |",
               position, byte_label(input_id), hex_byte(input_id), input_id,
               byte_label(target_id), hex_byte(target_id), target_id))
}

emit("", "**What the model learns:** at each input position it predicts a probability",
     "distribution over the 256 possible *next-byte tokens*. The target column is",
     "the observed next byte used to compute training loss; it is not a generated sample.",
     "A UTF-8 character can occupy multiple bytes, so one token need not equal one character.", "")

report <- paste(lines, collapse = "\n")
cat(report, "\n", sep = "")
if (nzchar(options$output)) {
  dir.create(dirname(options$output), recursive = TRUE, showWarnings = FALSE)
  writeLines(report, options$output, useBytes = TRUE)
  message("Saved Markdown report: ", options$output)
}
