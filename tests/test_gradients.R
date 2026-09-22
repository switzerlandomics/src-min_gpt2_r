# End-to-end finite-difference checks on a small, deterministic model.
# Run from repository root: Rscript tests/test_gradients.R
source("R/model.R")
model <- initialise_model(new_model_config(8L, 4L, 4L, 2L, 2L), seed = 42L)
inputs <- matrix(c(1L, 2L, 3L, 4L), nrow = 1L)
targets <- matrix(c(2L, 3L, 4L, 5L), nrow = 1L)
forward <- model_forward(model, inputs)
analytical <- model_backward(model, forward, targets)$gradients

read_path <- function(root, path) {
  for (key in path) root <- root[[key]]
  root
}
write_path <- function(root, path, value) {
  key <- path[[1L]]
  if (length(path) == 1L) root[[key]] <- value
  else root[[key]] <- write_path(root[[key]], path[-1L], value)
  root
}
check_parameter <- function(path, positions = 1:2, epsilon = 1e-5, tolerance = 5e-4) {
  original <- read_path(model$parameters, path)
  expected <- read_path(analytical, path)
  for (position in positions[positions <= length(original)]) {
    plus <- minus <- model
    plus_parameter <- minus_parameter <- original
    plus_parameter[position] <- plus_parameter[position] + epsilon
    minus_parameter[position] <- minus_parameter[position] - epsilon
    plus$parameters <- write_path(model$parameters, path, plus_parameter)
    minus$parameters <- write_path(model$parameters, path, minus_parameter)
    numerical <- (model_loss(plus, inputs, targets) -
                    model_loss(minus, inputs, targets)) / (2 * epsilon)
    relative_error <- abs(expected[position] - numerical) /
      max(1e-6, abs(expected[position]) + abs(numerical))
    if (!is.finite(relative_error) || relative_error > tolerance) {
      stop(sprintf("Gradient mismatch at %s[%d]: analytical=%g numerical=%g error=%g",
                   paste(unlist(path), collapse = "/"), position,
                   expected[position], numerical, relative_error))
    }
  }
}
paths <- list(
  list("token_embedding"), list("position_embedding"),
  list("blocks", 1L, "norm1", "scale"),
  list("blocks", 1L, "attention", "qkv", "weight"),
  list("blocks", 1L, "attention", "qkv", "bias"),
  list("blocks", 1L, "attention", "projection", "weight"),
  list("blocks", 1L, "norm2", "bias"),
  list("blocks", 1L, "feedforward", "up", "weight"),
  list("blocks", 1L, "feedforward", "down", "bias"),
  list("blocks", 2L, "attention", "qkv", "weight"),
  list("blocks", 2L, "feedforward", "down", "weight"),
  list("final_norm", "scale"))
for (path in paths) check_parameter(path)
cat("PASS: analytical versus numerical end-to-end gradients\n")
