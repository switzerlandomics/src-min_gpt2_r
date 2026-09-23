# Optional, model-derived figures. The numerical model has no dependency on ggplot2.
# Live monitoring writes only PNG. Detailed figures use the same ggplot object
# for compact PNG and (when svglite is installed) editable SVG exports.

# plot_colours <- list(
#   training = "#687984", validation = "#087A92", selection = "#E5262F",
#   attention_low = "#F1F8FA", attention_high = "#087A92",
#   masked = "#F3F2F2", grid = "#E7EBEE", border = "#D9D9D9"
# )

accent_colour <- "#E5262F"

plot_colours <- list(
  training = "#687984",
  validation = accent_colour,
  selection = accent_colour,
  attention_low = "#FDEBED",
  attention_high = accent_colour,
  masked = "#F3F2F2",
  grid = "#E7EBEE",
  border = "#D9D9D9"
)

plotting_available <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("ggplot2 is not installed; optional figures were skipped.")
    return(FALSE)
  }
  TRUE
}

# All theme text is at least 12 pt in the exported file. Avoid shrinking the
# entire SVG on the webpage: export close to the intended displayed dimensions.
plot_theme <- function() {
  ggplot2::theme_bw(base_size = 12, base_family = "sans") +
    ggplot2::theme(
      text = ggplot2::element_text(family = "sans", size = 12),
      plot.title = ggplot2::element_text(size = 16, face = "bold"),
      plot.subtitle = ggplot2::element_text(size = 12),
      plot.caption = ggplot2::element_text(size = 12, hjust = 0),
      axis.title = ggplot2::element_text(size = 12),
      axis.text = ggplot2::element_text(size = 12),
      legend.title = ggplot2::element_text(size = 12),
      legend.text = ggplot2::element_text(size = 12),
      strip.text = ggplot2::element_text(size = 12),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(colour = plot_colours$grid,
                                                  linewidth = 0.3),
      panel.border = ggplot2::element_rect(colour = plot_colours$border),
      legend.position = "top", plot.title.position = "plot",
      plot.margin = ggplot2::margin(8, 8, 8, 8)
    )
}

# Input 'path' is a PNG path. SVG uses the same plot object and physical size.
# For live monitoring keep svg = FALSE so figure updates remain lightweight.
save_plot <- function(plot, path, width, height, svg = FALSE, dpi = 240) {
  if (!grepl("\\.png$", path, ignore.case = TRUE))
    stop("save_plot() expects a PNG destination.", call. = FALSE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write_graphic <- function(destination, device = NULL) {
    extension <- if (is.null(device)) ".png" else ".svg"
    temporary <- tempfile(".figure-", tmpdir = dirname(destination), fileext = extension)
    on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
    arguments <- list(filename = temporary, plot = plot, width = width,
                      height = height, units = "in", bg = "white")
    if (is.null(device)) arguments$dpi <- dpi else arguments$device <- device
    do.call(ggplot2::ggsave, arguments)
    if (!file.rename(temporary, destination) &&
        !file.copy(temporary, destination, overwrite = TRUE))
      stop("Unable to save figure: ", destination, call. = FALSE)
  }
  write_graphic(path)
  if (svg && requireNamespace("svglite", quietly = TRUE)) {
    svg_path <- sub("\\.png$", ".svg", path, ignore.case = TRUE)
    # A missing optional SVG driver must not prevent the accompanying PNG.
    tryCatch(write_graphic(svg_path, svglite::svglite),
             error = function(error) warning("SVG export failed: ", conditionMessage(error)))
  }
  invisible(TRUE)
}

# A figure must never interrupt the training run.
try_plot <- function(description, expression) {
  tryCatch(force(expression), error = function(error) {
    warning("Could not create ", description, ": ", conditionMessage(error))
    invisible(FALSE)
  })
}

# Individual byte tokens may not be valid Unicode characters. Display ASCII
# where safe and a hexadecimal value otherwise; do not call them characters.
byte_label <- function(token_id, tokenizer, short = FALSE) {
  value <- as.integer(decode_bytes(token_id, tokenizer))
  readable <- switch(as.character(value), "9" = "\\t", "10" = "\\n",
                     "13" = "\\r", "32" = "space", NULL)
  if (is.null(readable)) readable <- if (value >= 33L && value <= 126L)
    intToUtf8(value) else sprintf("%02X", value)
  if (short) readable else sprintf("%s [%d]", readable, token_id)
}

token_label <- byte_label  # Keep a familiar name for earlier inspection scripts.

learning_plot_data <- function(metrics) {
  data <- rbind(
    data.frame(update = metrics$update, loss = metrics$training_loss,
               series = "Training window"),
    data.frame(update = metrics$update, loss = metrics$validation_loss,
               series = "Held-out validation")
  )
  data <- data[is.finite(data$loss), , drop = FALSE]
  data$series <- factor(data$series, levels = c("Training window", "Held-out validation"))
  data
}

plot_learning <- function(metrics, path, best_update = NULL,
                          best_loss = NULL, detailed = FALSE, svg = FALSE) {
  if (!plotting_available()) return(invisible(FALSE))
  data <- learning_plot_data(metrics)
  if (!nrow(data)) return(invisible(FALSE))
  plot <- ggplot2::ggplot(data, ggplot2::aes(update, loss, colour = series)) +
    ggplot2::geom_line(linewidth = 0.85, na.rm = TRUE) +
    ggplot2::geom_point(size = 1.7) +
    ggplot2::scale_colour_manual(values = c("Training window" = plot_colours$training,
                                            "Held-out validation" = plot_colours$validation)) +
    ggplot2::scale_x_continuous(labels = function(x) format(x, big.mark = ",", trim = TRUE)) +
    ggplot2::labs(title = if (detailed) "Learning history" else "Training and validation",
                  x = "Parameter updates", y = "Loss (nats/token)", colour = NULL) +
    plot_theme()
  if (!is.null(best_update) && length(best_update) == 1L && is.finite(best_update)) {
    plot <- plot + ggplot2::geom_vline(xintercept = best_update, linetype = "dashed",
                                      colour = plot_colours$selection, linewidth = 0.5)
    if (!is.null(best_loss) && length(best_loss) == 1L && is.finite(best_loss)) {
      plot <- plot + ggplot2::geom_point(data = data.frame(update = best_update, loss = best_loss),
        mapping = ggplot2::aes(update, loss), inherit.aes = FALSE,
        colour = plot_colours$selection, size = 2.4)
      if (detailed) plot <- plot + ggplot2::labs(
        caption = sprintf("Best validation: %.3f at update %s.", best_loss,
                          format(best_update, big.mark = ",")))
    }
  }
  save_plot(plot, path, width = 5.5, height = if (detailed) 4.5 else 4,
            svg = svg)
}

update_live_plots <- function(metrics, directory) {
  try_plot("live training figure",
           plot_learning(metrics, file.path(directory, "training.png")))
  invisible(TRUE)
}

# The model inspection array is indexed as [batch][head]; plots use batch 1.
# Capture the forward pass once for detailed figures and reuse across heads.
attention_plot_data <- function(model, token_ids, layer = 1L,
                                heads = NULL, inspection = NULL) {
  if (!length(token_ids) || length(token_ids) > model$config$context_length)
    stop("Inspection prompt has an invalid length.", call. = FALSE)
  if (length(layer) != 1L || is.na(layer) || layer != floor(layer) ||
      layer < 1L || layer > model$config$n_layers)
    stop("Invalid block number.", call. = FALSE)
  if (is.null(heads)) heads <- seq_len(model$config$n_heads)
  if (!length(heads) || anyNA(heads) || any(heads != floor(heads)) ||
      any(heads < 1L | heads > model$config$n_heads))
    stop("Invalid head number.", call. = FALSE)
  if (is.null(inspection)) inspection <- model_forward(
    model, matrix(token_ids, nrow = 1L), inspect = TRUE)$inspection
  saved <- inspection[[layer]]$attention
  n <- length(token_ids)
  rows <- lapply(heads, function(head) {
    weights <- saved[[head]]$weights
    if (!identical(dim(weights), c(n, n)))
      stop("Unexpected attention matrix dimensions.", call. = FALSE)
    data <- expand.grid(query = seq_len(n), key = seq_len(n))
    data$weight <- weights[cbind(data$query, data$key)]
    data$weight[data$key > data$query] <- NA_real_  # Mask is not observed weight zero.
    data$head <- sprintf("Head %d", head)
    data
  })
  do.call(rbind, rows)
}

attention_figure <- function(data, token_ids, tokenizer, layer, head) {
  n <- length(token_ids)
  labels <- vapply(token_ids, byte_label, character(1L), tokenizer = tokenizer,
                   short = TRUE)
  ggplot2::ggplot(data, ggplot2::aes(key, query, fill = weight)) +
    ggplot2::geom_tile(colour = plot_colours$border, linewidth = 0.3) +
    ggplot2::scale_fill_gradient(low = plot_colours$attention_low,
                                 high = plot_colours$attention_high,
                                 na.value = plot_colours$masked, limits = c(0, 1),
                                 breaks = seq(0, 1, by = 0.25),
                                 labels = sprintf("%.2f", seq(0, 1, by = 0.25)),
                                 name = "Weight",
                                 guide = ggplot2::guide_colourbar(
                                   direction = "vertical",
                                   title.position = "top",
                                   barheight = grid::unit(1.5, "in"),
                                   barwidth = grid::unit(0.16, "in")
                                 )) +
    ggplot2::scale_x_continuous(breaks = seq_len(n), labels = labels,
                                expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::scale_y_reverse(breaks = seq_len(n), labels = labels,
                             expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::coord_equal() +
    ggplot2::labs(title = sprintf("Block %d, head %d", layer, head),
                  x = "Key token", y = "Query token",
                  caption = "Grey: masked future positions.") +
    plot_theme() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      # Override the shared top legend for compact single-head attention plots.
      legend.position = "right",
      legend.direction = "vertical"
    )
}

# A compact single-head PNG for the main output folder; --plot-detailed writes
# the same kind of figure as one PNG + SVG pair for every layer and head.
plot_attention <- function(model, token_ids, layer = 1L, head = 1L, path,
                           tokenizer = NULL, svg = FALSE, inspection = NULL) {
  if (!plotting_available()) return(invisible(FALSE))
  if (is.null(tokenizer)) tokenizer <- new_byte_tokenizer()
  data <- attention_plot_data(model, token_ids, layer, head, inspection = inspection)
  save_plot(attention_figure(data, token_ids, tokenizer, layer, head),
            path, width = 5, height = 4.5, svg = svg)
}

next_token_probabilities <- function(model, prompt_ids) {
  if (!length(prompt_ids)) stop("Provide at least one input token.", call. = FALSE)
  context <- tail(prompt_ids, model$config$context_length)
  forward <- model_forward(model, matrix(context, nrow = 1L))
  logits <- as.numeric(forward$logits[1L, length(context), ])
  exp_scores <- exp(logits - max(logits))
  exp_scores / sum(exp_scores)
}

plot_next_token <- function(best, initial, prompt_ids, tokenizer, path,
                            top_n = 10L, svg = FALSE) {
  if (!plotting_available()) return(invisible(FALSE))
  best_probs <- next_token_probabilities(best$model, prompt_ids)
  selected <- head(order(best_probs, decreasing = TRUE), min(top_n, length(best_probs)))
  labels <- vapply(selected, byte_label, character(1L), tokenizer = tokenizer)
  rows <- list(data.frame(token = labels, probability = best_probs[selected],
                          checkpoint = "Best validation"))
  if (!is.null(initial)) {
    initial_probs <- next_token_probabilities(initial$model, prompt_ids)
    rows[[2L]] <- data.frame(token = labels, probability = initial_probs[selected],
                             checkpoint = "Before training")
  }
  data <- do.call(rbind, rows)
  data$token <- factor(data$token, levels = rev(labels))
  plot <- ggplot2::ggplot(data, ggplot2::aes(probability, token, fill = checkpoint)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8), width = 0.7) +
    ggplot2::scale_fill_manual(values = c("Best validation" = plot_colours$validation,
                                         "Before training" = plot_colours$training)) +
    ggplot2::labs(title = "Next-token probabilities", x = "Probability", y = "Byte token",
                  fill = NULL) +
    plot_theme()
  save_plot(plot, path, width = 5, height = 4.5, svg = svg)
}

create_detailed_plots <- function(best, initial, metrics, prompt_ids, tokenizer, directory) {
  if (!plotting_available()) return(invisible(FALSE))
  if (!requireNamespace("svglite", quietly = TRUE))
    message("svglite is not installed; detailed PNGs will be created without SVGs.")
  detailed_dir <- file.path(directory, "detailed")
  dir.create(detailed_dir, recursive = TRUE, showWarnings = FALSE)
  try_plot("detailed learning history", plot_learning(
    metrics, file.path(detailed_dir, "training_detail.png"),
    best_update = best$update, best_loss = best$validation_loss,
    detailed = TRUE, svg = TRUE))
  try_plot("next-token probabilities", plot_next_token(
    best, initial, prompt_ids, tokenizer,
    file.path(detailed_dir, "next_token_probabilities.png"), svg = TRUE))

  # One inspection of the same prompt, reused for every layer/head figure.
  inspection <- tryCatch(model_forward(
    best$model, matrix(prompt_ids, nrow = 1L), inspect = TRUE)$inspection,
    error = function(error) {
      warning("Could not inspect model attention: ", conditionMessage(error))
      NULL
    })
  if (is.null(inspection)) return(invisible(FALSE))
  for (layer in seq_len(best$model$config$n_layers)) {
    for (head in seq_len(best$model$config$n_heads)) {
      path <- file.path(detailed_dir, sprintf("attention_block_%02d_head_%02d.png", layer, head))
      try_plot(sprintf("attention block %d head %d", layer, head),
               plot_attention(best$model, prompt_ids, layer = layer, head = head,
                              tokenizer = tokenizer, path = path, svg = TRUE,
                              inspection = inspection))
    }
  }
  invisible(TRUE)
}
