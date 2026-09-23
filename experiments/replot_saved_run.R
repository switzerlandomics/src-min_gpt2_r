#!/usr/bin/env Rscript

# Regenerate figures from a completed experiment without training or writing
# to its model files, checkpoint, metrics, samples, or original plots.
# Run from the repository root:
#   Rscript experiments/replot_saved_run.R output/20260922_184933

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L || !nzchar(arguments[1L])) {
  stop("Usage: Rscript experiments/replot_saved_run.R output/RUN_DIRECTORY",
       call. = FALSE)
}

source("R/tokenizer.R")
source("R/data.R")
source("R/model.R")
source("R/plots.R")

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2 to regenerate plots: install.packages('ggplot2')",
       call. = FALSE)
}

run_dir <- arguments[1L]
required <- file.path(run_dir, c("best_model.rds", "latest_checkpoint.rds", "metrics.csv"))
missing <- required[!file.exists(required)]
if (length(missing)) {
  stop("Missing run file(s): ", paste(missing, collapse = ", "), call. = FALSE)
}

best <- readRDS(file.path(run_dir, "best_model.rds"))
checkpoint <- readRDS(file.path(run_dir, "latest_checkpoint.rds"))
metrics <- utils::read.csv(file.path(run_dir, "metrics.csv"))
tokenizer <- best$tokenizer
if (is.null(tokenizer)) tokenizer <- checkpoint$tokenizer
if (is.null(tokenizer)) stop("No tokenizer in the saved run.", call. = FALSE)

# Reconstruct the exact inspection prompt from the original validation split.
corpus_path <- checkpoint$corpus_path
if (is.null(corpus_path) || !file.exists(corpus_path)) {
  stop("The training corpus is not present at its recorded path: ", corpus_path,
       "\nRestore it there before regenerating the plots.", call. = FALSE)
}
checksum <- unname(tools::md5sum(corpus_path))
if (is.null(checkpoint$corpus_checksum) ||
    !identical(checksum, checkpoint$corpus_checksum)) {
  stop("The training corpus differs from the original run; refusing to use a different inspection prompt.",
       call. = FALSE)
}

splits <- split_corpus(read_corpus(corpus_path, tokenizer))
prompt_ids <- head(splits$validation, min(8L, best$model$config$context_length))

initial_path <- file.path(run_dir, "initial_model.rds")
initial <- if (file.exists(initial_path)) readRDS(initial_path) else NULL

# This destination is separate: existing checkpoints, data, and original figures
# remain untouched. Outputs are under figures_refreshed/detailed/.
output_dir <- file.path(run_dir, "figures_refreshed")
cat("Regenerating detailed figures using saved best model at update ",
    best$update, ".\n", sep = "")
cat("Output directory: ", output_dir, "\n", sep = "")

create_detailed_plots(
  best = best,
  initial = initial,
  metrics = metrics,
  prompt_ids = prompt_ids,
  tokenizer = tokenizer,
  directory = output_dir
)
plot_attention(
  best$model,
  prompt_ids,
  tokenizer = tokenizer,
  path = file.path(output_dir, "attention.png"),
  svg = TRUE
)

cat("Finished. Original model files and figures were not modified.\n")
