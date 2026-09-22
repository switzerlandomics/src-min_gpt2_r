#!/usr/bin/env Rscript

# Reproducible GPT-2-style language-model experiment.
# Run from the repository root.
#
# The model and its gradients are implemented in R/model.R.
# Plotting is optional and remains separate from the training calculations.

source("R/tokenizer.R")
source("R/data.R")
source("R/model.R")
source("R/optimiser.R")
source("R/evaluation.R")
source("R/progress.R")
source("R/plots.R")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

parse_options <- function(arguments) {

  options <- list(
    input = "data/input.txt",
    iterations = 100L,
    context_length = 16L,
    embedding_size = 24L,
    n_heads = 4L,
    n_layers = 2L,
    batch_size = 2L,
    learning_rate = 0.002,
    seed = 42L,
    validation_interval = 25L,
    checkpoint_interval = 50L,
    sample_tokens = 60L,
    resume = "",
    plot = FALSE,
    plot_detailed = FALSE
  )

  types <- lapply(options, typeof)

  for (argument in arguments) {

    if (argument == "--plot") {
      options$plot <- TRUE
      next
    }

    if (argument == "--no-plot") {
      options$plot <- FALSE
      options$plot_detailed <- FALSE
      next
    }

    if (argument == "--plot-detailed") {
      options$plot <- TRUE
      options$plot_detailed <- TRUE
      next
    }

    if (argument == "--no-plot-detailed") {
      options$plot_detailed <- FALSE
      next
    }

    if (argument == "--help") {

      cat(
        "Usage: Rscript experiments/run.R [--name=value] [plotting flags]\n\n",
        "Options:\n",
        "  ", paste(names(options), collapse = ", "), "\n\n",
        "Plotting flags:\n",
        "  --plot             Refresh training.png during training.\n",
        "  --plot-detailed    Enable --plot and create detailed final figures.\n",
        "  --no-plot          Disable all plotting.\n",
        "  --no-plot-detailed Disable detailed figures.\n\n",
        "On resume, only --iterations and plotting flags may be changed.\n",
        "--iterations is the total target, not additional updates.\n",
        sep = ""
      )

      quit(save = "no", status = 0L)
    }

    matches <- regmatches(
      argument,
      regexec("^--([a-z][a-z0-9-]*)=(.*)$", argument)
    )[[1L]]

    if (length(matches) != 3L) {
      stop("Unknown argument: ", argument, call. = FALSE)
    }

    key <- gsub("-", "_", matches[2L])

    if (!key %in% names(options) ||
        key %in% c("plot", "plot_detailed")) {
      stop("Unknown option: ", key, call. = FALSE)
    }

    value <- matches[3L]

    if (types[[key]] == "integer") {

      value <- suppressWarnings(as.integer(value))

      if (is.na(value)) {
        stop("Expected an integer for ", key, call. = FALSE)
      }

    } else if (types[[key]] == "double") {

      value <- suppressWarnings(as.numeric(value))

      if (!is.finite(value)) {
        stop("Expected a finite number for ", key, call. = FALSE)
      }
    }

    options[[key]] <- value
  }

  if (options$iterations < 0L ||
      options$batch_size < 1L ||
      options$validation_interval < 1L ||
      options$checkpoint_interval < 1L ||
      options$sample_tokens < 0L) {
    stop("Invalid run setting.", call. = FALSE)
  }

  options
}


arguments <- commandArgs(trailingOnly = TRUE)
options <- parse_options(arguments)

original_command <- paste(
  c("Rscript experiments/run.R", shQuote(arguments)),
  collapse = " "
)

resuming <- nzchar(options$resume)

if (resuming) {

  allowed <- grepl(
    "^--(resume|iterations)=|^--(no-)?plot(-detailed)?$",
    arguments
  )

  if (any(!allowed)) {
    stop(
      "Resume accepts only --resume, --iterations and plotting flags.",
      call. = FALSE
    )
  }

  if (!any(grepl("^--iterations=", arguments))) {
    stop(
      "Specify the total --iterations when resuming.",
      call. = FALSE
    )
  }
}


# ---------------------------------------------------------------------------
# Checkpoint and metrics saving
# ---------------------------------------------------------------------------

# Atomic write_rds() and write_csv() live in R/progress.R. There is one
# implementation of disk replacement and none in the numerical model.

save_state <- function(directory, model, optimiser, step,
                       best_loss, best_update, metrics, config,
                       tokenizer, corpus_path, corpus_checksum,
                       run_options, samples) {

  state <- list(
    model = model,
    optimiser = optimiser,
    step = step,
    best_loss = best_loss,
    best_update = best_update,
    metrics = metrics,
    config = config,
    tokenizer = tokenizer,
    corpus_path = corpus_path,
    corpus_checksum = corpus_checksum,
    run_options = run_options,
    samples = samples,
    rng = .Random.seed
  )

  write_rds(
    state,
    file.path(directory, "latest_checkpoint.rds")
  )
}


save_best <- function(directory, model, tokenizer, update,
                      validation_loss, corpus_checksum) {

  best <- list(
    model = model,
    tokenizer = tokenizer,
    update = update,
    validation_loss = validation_loss,
    corpus_checksum = corpus_checksum
  )

  write_rds(
    best,
    file.path(directory, "best_model.rds")
  )
}


record_command <- function(directory, command, options,
                           corpus_path, append = FALSE) {

  lines <- c(
    paste("Original command:", command),
    paste("Corpus:", corpus_path),
    paste(
      "Resolved options:",
      paste(capture.output(dput(options)), collapse = "")
    ),
    paste("R version:", R.version.string)
  )

  if (append) {
    lines <- c(
      "",
      paste("Resume:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
      lines
    )
  }

  cat(
    paste(lines, collapse = "\n"),
    "\n",
    file = file.path(directory, "run_command.txt"),
    append = append
  )
}


# ---------------------------------------------------------------------------
# Model-derived text samples
# ---------------------------------------------------------------------------

# All recorded generations use a fixed prompt, seed and sampling settings.
# Generation uses with_local_seed() inside generate_tokens(), so taking a
# sample does not change the random sequence used for training batches.
sample_seed <- 100L
sample_temperature <- 1
sample_top_k <- 20L

make_sample <- function(model, tokenizer, prompt_ids, update,
                        validation_loss, n_tokens) {
  generated <- generate_tokens(
    model, prompt_ids, n_tokens,
    temperature = sample_temperature,
    top_k = sample_top_k,
    seed = sample_seed
  )
  list(
    update = as.integer(update),
    validation_loss = as.numeric(validation_loss),
    prompt_ids = as.integer(prompt_ids),
    n_tokens = as.integer(n_tokens),
    temperature = sample_temperature,
    top_k = sample_top_k,
    seed = sample_seed,
    text = decode_tokens(generated, tokenizer, incomplete = "replace")
  )
}

format_sample <- function(sample, tokenizer) {
  prompt <- decode_tokens(sample$prompt_ids, tokenizer, incomplete = "replace")
  paste0(
    sprintf(
      '=== update %d | validation %.6f nats/token | prompt %s | temperature %s | top_k %d | seed %d ===',
      sample$update, sample$validation_loss,
      encodeString(prompt, quote = '"'),
      format(sample$temperature), sample$top_k, sample$seed
    ),
    "\n", sample$text
  )
}

# The checkpoint's sample list is the authoritative history. Rebuild the
# text file from it on resume rather than appending a potentially duplicated
# sample after an interrupted update.
write_samples <- function(samples, tokenizer, directory) {
  if (!length(samples)) return(invisible(FALSE))
  text <- paste(
    vapply(samples, format_sample, character(1L), tokenizer = tokenizer),
    collapse = "\n\n"
  )
  atomic_write(
    file.path(directory, "samples.txt"),
    function(temporary) writeLines(text, temporary, useBytes = TRUE)
  )
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Initialise or resume
# ---------------------------------------------------------------------------

if (resuming) {

  directory <- options$resume

  checkpoint <- readRDS(
    file.path(directory, "latest_checkpoint.rds")
  )

  requested_iterations <- options$iterations
  requested_plot <- options$plot
  requested_plot_detailed <- options$plot_detailed

  model <- checkpoint$model
  optimiser <- checkpoint$optimiser
  tokenizer <- checkpoint$tokenizer
  config <- checkpoint$config
  metrics <- checkpoint$metrics
  samples <- if (is.null(checkpoint$samples)) list() else checkpoint$samples

  step <- checkpoint$step
  best_loss <- checkpoint$best_loss
  corpus_path <- checkpoint$corpus_path

  options <- checkpoint$run_options

  # Support checkpoints produced before --plot-detailed was introduced.
  if (is.null(options$plot_detailed)) {
    options$plot_detailed <- FALSE
  }

  options$iterations <- requested_iterations
  options$plot <- requested_plot
  options$plot_detailed <- requested_plot_detailed
  options$resume <- directory

  if (options$iterations < step) {
    stop(
      "iterations cannot be below the saved checkpoint step.",
      call. = FALSE
    )
  }

  best_path <- file.path(directory, "best_model.rds")

  if (!file.exists(best_path)) {
    stop(
      "best_model.rds is missing from the selected run.",
      call. = FALSE
    )
  }

  saved_best <- readRDS(best_path)

  # Older runs did not record best_update in their resumable checkpoint.
  best_update <- checkpoint$best_update

  if (is.null(best_update)) {
    best_update <- saved_best$update
  }

  # In older runs, best_model.rds may be newer than the last resumable
  # checkpoint. Retain that genuine measured result, but make the
  # discrepancy explicit.
  if (saved_best$update > step) {

    warning(
      "The saved best model is newer than the resumable checkpoint. ",
      "This can occur in runs made with the previous runner. ",
      "Training will resume from the checkpoint while preserving ",
      "the existing best-validation result."
    )

    best_loss <- saved_best$validation_loss
    best_update <- saved_best$update

  } else if (saved_best$update < best_update) {

    # With the updated runner, a checkpoint is written before a newly
    # improved best_model.rds. If interrupted between those writes,
    # the checkpoint itself contains the best model at that update.
    if (!identical(best_update, step)) {
      stop(
        "Best-model metadata are inconsistent with the checkpoint.",
        call. = FALSE
      )
    }

    save_best(
      directory,
      model,
      tokenizer,
      step,
      best_loss,
      checkpoint$corpus_checksum
    )

  } else if (!isTRUE(all.equal(
    saved_best$validation_loss,
    best_loss,
    tolerance = 1e-10
  ))) {

    stop(
      "Best-model loss does not match the resumable checkpoint.",
      call. = FALSE
    )
  }

  if (!is.null(checkpoint$rng)) {
    assign(
      ".Random.seed",
      checkpoint$rng,
      envir = .GlobalEnv
    )
  } else {
    stop(
      "Checkpoint is missing its random-number state.",
      call. = FALSE
    )
  }

  cat(
    "Resuming ", directory,
    " at update ", step, "\n",
    sep = ""
  )

} else {

  set.seed(options$seed)

  tokenizer <- new_byte_tokenizer()

  config <- new_model_config(
    vocabulary_size(tokenizer),
    options$context_length,
    options$embedding_size,
    options$n_heads,
    options$n_layers
  )

  model <- initialise_model(config)
  optimiser <- adam_initialise(model$parameters)

  metrics <- data.frame(
    update = integer(),
    training_loss = numeric(),
    validation_loss = numeric(),
    gradient_norm = numeric()
  )
  samples <- list()

  step <- 0L
  best_loss <- Inf
  best_update <- NA_integer_

  corpus_path <- normalizePath(
    options$input,
    mustWork = TRUE
  )

  directory <- file.path(
    "output",
    format(Sys.time(), "%Y%m%d_%H%M%S")
  )

  if (dir.exists(directory)) {
    stop(
      "Output directory already exists; retry.",
      call. = FALSE
    )
  }

  dir.create(directory, recursive = TRUE)

  cat(
    sprintf(
      "New experiment %s (%d parameters)\n",
      directory,
      parameter_count(model)
    )
  )
}


# Start a new session clock after the model/checkpoint is loaded. The ETA
# is based on updates performed in this session, not on earlier run history.
session_start <- proc.time()[["elapsed"]]
session_start_step <- step
logger <- make_logger(file.path(directory, "experiment.log"))
logger("INFO", "Session started at update %d; target %d", step, options$iterations)

# ---------------------------------------------------------------------------
# Load and validate the corpus
# ---------------------------------------------------------------------------

tokens <- read_corpus(corpus_path, tokenizer)
checksum <- unname(tools::md5sum(corpus_path))

if (resuming &&
    !identical(checksum, checkpoint$corpus_checksum)) {

  stop(
    "Corpus changed since checkpoint; refusing to resume.",
    call. = FALSE
  )
}

splits <- split_corpus(tokens)

for (name in names(splits)) {

  if (length(splits[[name]]) <= config$context_length) {

    stop(
      "The ", name,
      " split must exceed context_length; use more data or less context.",
      call. = FALSE
    )
  }
}


# Use the same short, held-out prompt for every periodic and final sample.
prompt_ids <- head(splits$validation, min(8L, config$context_length))

if (resuming) {

  # The checkpoint is authoritative for recoverable training history.
  # Restore metrics.csv if an interruption occurred during file updates.
  write_csv(metrics, file.path(directory, "metrics.csv"))
  write_samples(samples, tokenizer, directory)

  record_command(
    directory,
    original_command,
    options,
    corpus_path,
    append = TRUE
  )

} else {

  writeLines(
    paste("Corpus MD5:", checksum),
    file.path(directory, "corpus_checksum.txt")
  )

  record_command(
    directory,
    original_command,
    options,
    corpus_path
  )

  # Save the actual initial parameters rather than relying on a seed
  # to reconstruct them after the implementation has changed.
  write_rds(
    list(
      model = model,
      tokenizer = tokenizer,
      corpus_checksum = checksum
    ),
    file.path(directory, "initial_model.rds")
  )
}


# A legacy checkpoint may predate samples.txt. If its actual initial model
# was saved, an update-zero sample can be reproduced from those parameters;
# no intermediate samples are inferred from later checkpoints.
if (resuming && !length(samples) &&
    file.exists(file.path(directory, "initial_model.rds"))) {
  initial_row <- which(metrics$update == 0L & is.finite(metrics$validation_loss))
  if (length(initial_row)) {
    initial <- readRDS(file.path(directory, "initial_model.rds"))
    if (identical(initial$corpus_checksum, checksum)) {
      samples[[1L]] <- make_sample(
        initial$model, tokenizer, prompt_ids, 0L,
        metrics$validation_loss[initial_row[1L]], options$sample_tokens
      )
      write_samples(samples, tokenizer, directory)
    }
  }
}

# ---------------------------------------------------------------------------
# Validation and monitoring
# ---------------------------------------------------------------------------

record_validation <- function(update, training_loss,
                              validation_loss, gradient_norm) {

  metrics <<- rbind(
    metrics,
    data.frame(
      update = update,
      training_loss = training_loss,
      validation_loss = validation_loss,
      gradient_norm = gradient_norm
    )
  )

  improved <- validation_loss < best_loss

  if (improved) {
    best_loss <<- validation_loss
    best_update <<- update
  }

  # Sample only at update zero, checkpoint updates, and the final target.
  # The validation figure can update more frequently without producing
  # thousands of generated-text records or slowing down every measurement.
  should_sample <- update == 0L ||
    update %% options$checkpoint_interval == 0L ||
    update == options$iterations

  if (should_sample &&
      !any(vapply(samples, function(item) item$update == update, logical(1L)))) {
    samples[[length(samples) + 1L]] <<- make_sample(
      model, tokenizer, prompt_ids, update,
      validation_loss, options$sample_tokens
    )
  }

  # Save the complete recoverable state at every validation. This ensures
  # metrics and model-selection metadata correspond to the same update.
  save_state(
    directory,
    model,
    optimiser,
    update,
    best_loss,
    best_update,
    metrics,
    config,
    tokenizer,
    corpus_path,
    checksum,
    options,
    samples
  )

  # Write the selected model after the checkpoint. If interrupted here,
  # a subsequent resume can recover it from the checkpoint.
  if (improved) {

    save_best(
      directory,
      model,
      tokenizer,
      update,
      validation_loss,
      checksum
    )
  }

  write_csv(metrics, file.path(directory, "metrics.csv"))
  if (should_sample) write_samples(samples, tokenizer, directory)

  if (options$plot) {

    update_live_plots(
      metrics,
      directory
    )
  }

  invisible(improved)
}


# An initial validation point is recorded only for a new experiment.
# Resuming update zero must not create a duplicate measurement.

if (!resuming) {

  validation <- evaluate_model(
    model,
    splits$validation
  )

  record_validation(
    update = 0L,
    training_loss = NA_real_,
    validation_loss = validation,
    gradient_norm = NA_real_
  )

  logger("INFO", "Update 0: validation %.4f nats/token", validation)
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if (options$iterations > step) {

  for (update in seq.int(
    step + 1L,
    options$iterations
  )) {

    batch <- make_batch(
      splits$train,
      config$context_length,
      options$batch_size
    )

    forward <- model_forward(
      model,
      batch$token_ids
    )

    backward <- model_backward(
      model,
      forward,
      batch$targets
    )

    updated <- adam_update(
      model$parameters,
      backward$gradients,
      optimiser,
      learning_rate = options$learning_rate,
      clip_norm = 1
    )

    model$parameters <- updated$parameters
    optimiser <- updated$state

    # Checkpoint samples need an actual validation measurement of the
    # sampled parameters, even when the two configured intervals differ.
    should_validate <-
      update %% options$validation_interval == 0L ||
      update %% options$checkpoint_interval == 0L ||
      update == options$iterations

    if (should_validate) {

      validation <- evaluate_model(
        model,
        splits$validation
      )

      record_validation(
        update = update,
        training_loss = backward$loss,
        validation_loss = validation,
        gradient_norm = updated$gradient_norm
      )

      elapsed <- proc.time()[["elapsed"]] - session_start
      status <- progress_status(
        completed = update, total = options$iterations, elapsed = elapsed,
        active_completed = update - session_start_step,
        active_elapsed = elapsed
      )
      logger("INFO", "Update %d/%d: train %.4f | validation %.4f | best %.4f | %s",
             update, options$iterations, backward$loss, validation, best_loss, status)
    }

    # Additional checkpoints between validations are permitted. A step
    # already saved by record_validation() does not need saving again.
    should_checkpoint <-
      update %% options$checkpoint_interval == 0L ||
      update == options$iterations

    if (should_checkpoint && !should_validate) {

      save_state(
        directory,
        model,
        optimiser,
        update,
        best_loss,
        best_update,
        metrics,
        config,
        tokenizer,
        corpus_path,
        checksum,
        options,
        samples
      )
    }
  }
}


# ---------------------------------------------------------------------------
# Final outputs
# ---------------------------------------------------------------------------

best <- readRDS(
  file.path(directory, "best_model.rds")
)

best_record <- make_sample(
  best$model, tokenizer, prompt_ids, best$update,
  best$validation_loss, options$sample_tokens
)

writeLines(
  c(
    sprintf(
      "Best validation update: %d (%.6f nats/token)",
      best$update,
      best$validation_loss
    ),
    paste(
      "Prompt (raw bytes):",
      paste(
        decode_bytes(prompt_ids, tokenizer),
        collapse = " "
      )
    ),
    "Generated text:",
    best_record$text
  ),
  file.path(directory, "sample.txt"),
  useBytes = TRUE
)


# Compare the actual update-zero model with the best-validation checkpoint.
# The best model may have been selected at a validation update that was not a
# scheduled text-sample update, so generate it directly from best_model.rds.
initial_record <- Filter(function(item) item$update == 0L, samples)
comparison <- c(
  "Text generation: initial and best-validation model",
  "Both samples use identical prompt, length, temperature, top_k and seed.",
  "The selected model was chosen by validation loss, not sample readability.",
  ""
)
if (length(initial_record)) {
  comparison <- c(comparison, format_sample(initial_record[[1L]], tokenizer), "")
} else {
  comparison <- c(comparison,
    "Initial sample unavailable: the run did not preserve its initial model.", "")
}
comparison <- c(comparison, format_sample(best_record, tokenizer))
atomic_write(
  file.path(directory, "generation_comparison.txt"),
  function(temporary) writeLines(comparison, temporary, useBytes = TRUE)
)


if (options$plot) {

  # Also works when resuming a completed run without additional training.
  update_live_plots(
    metrics,
    directory
  )

  tryCatch(
    plot_attention(
      best$model,
      prompt_ids,
      path = file.path(directory, "attention.png")
    ),
    error = function(error) {
      warning(
        "Could not produce attention.png: ",
        conditionMessage(error)
      )
    }
  )
}


if (options$plot_detailed) {

  initial_path <- file.path(
    directory,
    "initial_model.rds"
  )

  initial <- if (file.exists(initial_path)) {
    readRDS(initial_path)
  } else {
    NULL
  }

  create_detailed_plots(
    best = best,
    initial = initial,
    metrics = metrics,
    prompt_ids = prompt_ids,
    tokenizer = tokenizer,
    directory = directory
  )
}


logger("INFO", "Finished. Results: %s", directory)

