# Run from repository root:
# Rscript tests/test_training.R

source("R/model.R")
source("R/optimiser.R")
source("R/evaluation.R")


# ---------------------------------------------------------------------------
# Tiny-data learning
# ---------------------------------------------------------------------------

model <- initialise_model(
  new_model_config(
    vocab_size = 8L,
    context_length = 8L,
    embedding_size = 8L,
    n_heads = 2L,
    n_layers = 2L
  ),
  seed = 7L
)

inputs <- matrix(
  c(1L, 2L, 3L, 4L, 5L),
  nrow = 1L
)

targets <- matrix(
  c(2L, 3L, 4L, 5L, 6L),
  nrow = 1L
)

initial_loss <- model_loss(
  model,
  inputs,
  targets
)

state <- adam_initialise(
  model$parameters
)

for (step in seq_len(60L)) {

  forward <- model_forward(
    model,
    inputs
  )

  backward <- model_backward(
    model,
    forward,
    targets
  )

  update <- adam_update(
    model$parameters,
    backward$gradients,
    state,
    learning_rate = 0.01,
    clip_norm = 1
  )

  model$parameters <- update$parameters
  state <- update$state
}

final_loss <- model_loss(
  model,
  inputs,
  targets
)

stopifnot(
  is.finite(final_loss),
  final_loss < initial_loss * 0.7
)

stopifnot(
  identical(state$step, 60L)
)


# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------

ids_a <- generate_tokens(
  model,
  c(1L, 2L),
  5L,
  seed = 123L
)

ids_b <- generate_tokens(
  model,
  c(1L, 2L),
  5L,
  seed = 123L
)

stopifnot(
  identical(ids_a, ids_b),
  length(ids_a) == 7L
)

cat(
  sprintf(
    "PASS: tiny-data learning (loss %.4f -> %.4f) and deterministic generation\n",
    initial_loss,
    final_loss
  )
)


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

run_experiment <- function(arguments) {

  output <- suppressWarnings(
    system2(
      file.path(R.home("bin"), "Rscript"),
      args = shQuote(
        c("experiments/run.R", arguments)
      ),
      stdout = TRUE,
      stderr = TRUE
    )
  )

  status <- attr(output, "status")

  if (!is.null(status) && status != 0L) {

    stop(
      paste(
        c("Experiment runner failed:", output),
        collapse = "\n"
      ),
      call. = FALSE
    )
  }

  output
}


output <- run_experiment(
  c(
    "--input=data/input.txt",
    "--iterations=2",
    "--context-length=8",
    "--embedding-size=8",
    "--n-heads=2",
    "--n-layers=2",
    "--batch-size=1",
    "--validation-interval=1",
    "--checkpoint-interval=5",
    "--sample-tokens=0",
    "--plot"
  )
)

run_line <- grep(
  "^New experiment output/",
  output,
  value = TRUE
)

stopifnot(length(run_line) == 1L)

directory <- sub(
  "^New experiment (output/[^ ]+) \\(.*$",
  "\\1",
  run_line
)

stopifnot(dir.exists(directory))


# ---------------------------------------------------------------------------
# Saved metrics and checkpoint consistency
# ---------------------------------------------------------------------------

metrics <- read.csv(
  file.path(directory, "metrics.csv")
)

checkpoint <- readRDS(
  file.path(directory, "latest_checkpoint.rds")
)

best <- readRDS(
  file.path(directory, "best_model.rds")
)

initial <- readRDS(
  file.path(directory, "initial_model.rds")
)

stopifnot(
  identical(checkpoint$step, 2L),
  tail(metrics$update, 1L) == 2L,
  identical(metrics, checkpoint$metrics),
  identical(best$update, checkpoint$best_update),
  isTRUE(
    all.equal(
      best$validation_loss,
      checkpoint$best_loss,
      tolerance = 1e-12
    )
  ),
  !is.null(initial$model)
)


# ---------------------------------------------------------------------------
# Regenerate detailed figures without additional training
# ---------------------------------------------------------------------------

run_experiment(
  c(
    paste0("--resume=", directory),
    "--iterations=2",
    "--plot-detailed"
  )
)

after <- readRDS(
  file.path(directory, "latest_checkpoint.rds")
)

stopifnot(
  identical(after$step, 2L)
)

if (requireNamespace("ggplot2", quietly = TRUE)) {

  expected_figures <- c(
    "training.png",
    "attention.png",
    file.path("detailed", "training_detail.png"),
    file.path("detailed", "attention_block_01_head_01.png"),
    file.path("detailed", "attention_block_02_head_02.png"),
    file.path("detailed", "next_token_probabilities.png")
  )

  stopifnot(
    all(
      file.exists(
        file.path(directory, expected_figures)
      )
    )
  )
  if (requireNamespace("svglite", quietly = TRUE)) {
    expected_svg <- sub("\\.png$", ".svg", expected_figures[-c(1L, 2L)])
    stopifnot(all(file.exists(file.path(directory, expected_svg))))
  }
}


cat(
  "PASS: runner metrics, checkpoints, resumption and optional figures\n"
)

source("R/progress.R")
stopifnot(identical(format_duration(0), "0s"),
          grepl("ETA calculating", progress_status(0, 0, 0, 0, 0), fixed = TRUE))
