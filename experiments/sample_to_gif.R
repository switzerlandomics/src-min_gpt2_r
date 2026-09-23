#!/usr/bin/env Rscript

# Training-sample animation: static preview, then optional typing GIF with frame progress.
# install.packages(c("magick", "stringr"))
# Run from the same working directory used for the input/output paths below.

library(magick)
library(stringr)
library(grid)

input_path <- "../output_example/samples.txt"
preview_png <- "../output_example/sample_preview.png"
output_gif <- "../output_example/sample_typing_demo.gif"

# Set preview_only = TRUE while refining the static layout.
preview_only <- FALSE
preview_update <- 10000L
fast_test <- FALSE  # TRUE: animate only the initial, middle, and final samples.

# Animation settings. For a quick GIF test, use chars_per_frame = 20,
# fps = 20, hold_start_frames = 1, hold_end_frames = 3,
# and block_pause_frames = 1.
width <- 1100L
height <- 700L
fps <- 20L
chars_per_frame <- 3L
hold_start_frames <- 6L
hold_end_frames <- 12L
block_pause_frames <- 8L
max_line_chars <- 34L  # Wrap each complete sample once, before making any frames.

if (chars_per_frame < 1L || fps < 1L ||
    any(c(hold_start_frames, hold_end_frames, block_pause_frames) < 0L)) {
  stop("Invalid animation settings.")
}

# Visual settings; all positions are normalised to the same fixed canvas.
bg_colour <- "#F7F7F8"
card_colour <- "#FFFFFF"
border_colour <- "#E5E7EB"
text_colour <- "#111827"
muted_colour <- "#4B5563"
pill_fill <- "#EEF2F7"
# accent_colour <- "#087A92"  # Switzerland Omics teal
accent_colour <- "#E5262F"
mono_family <- "Courier"
sans_family <- "Helvetica"

read_text_file <- function(path) {
  paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
}

split_blocks <- function(text) {
  lines <- strsplit(gsub("\r\n?", "\n", text), "\n", fixed = TRUE)[[1L]]
  starts <- grep("^=== update [0-9]+[[:space:]]*\\|", lines)
  if (!length(starts)) stop("No '=== update ...' headers found in: ", input_path)
  ends <- c(starts[-1L] - 1L, length(lines))
  Map(function(start, end) paste(lines[start:end], collapse = "\n"), starts, ends)
}

parse_block <- function(block) {
  lines <- strsplit(block, "\n", fixed = TRUE)[[1L]]
  header <- lines[1L]
  m <- str_match(
    header,
    '^=== update\\s+([0-9]+)\\s+\\|\\s+validation\\s+([0-9.]+)\\s+nats/token\\s+\\|\\s+prompt\\s+"([^"]*)"\\s+\\|\\s+temperature\\s+([0-9.]+)\\s+\\|\\s+top_k\\s+([0-9]+)\\s+\\|\\s+seed\\s+([0-9]+)\\s+===$'
  )
  if (anyNA(m)) stop("Could not parse header:\n", header)
  list(
    update = as.integer(m[2L]),
    validation = as.numeric(m[3L]),
    prompt = m[4L],
    temperature = as.numeric(m[5L]),
    top_k = as.integer(m[6L]),
    seed = as.integer(m[7L]),
    body = sub("\n+$", "", sub("^\n+", "", paste(lines[-1L], collapse = "\n")))
  )
}

# Fixed-geometry metric: the label stays at the left and the value is
# right-aligned to a constant edge, regardless of how many digits it has.
draw_metric <- function(left, width, label, value) {
  centre <- left + width / 2
  grid.roundrect(
    x = centre, y = 0.884, width = width, height = 0.058,
    r = unit(7, "pt"),
    gp = gpar(fill = pill_fill, col = NA)
  )
  grid.text(
    label, x = left + 0.018, y = 0.884, just = c("left", "centre"),
    gp = gpar(col = muted_colour, fontsize = 10.5, fontfamily = sans_family)
  )
  grid.text(
    value, x = left + width - 0.018, y = 0.884,
    just = c("right", "centre"),
    gp = gpar(col = text_colour, fontsize = 12, fontfamily = mono_family,
              fontface = "bold")
  )
}

# Deliberately use the FULL history to fix both axes across all frames.
# Draw only measurements taken at/before the currently displayed update.
draw_validation_chart <- function(history, current_update) {
  grid.roundrect(
    x = 0.50, y = 0.222, width = 0.82, height = 0.318,
    r = unit(10, "pt"),
    gp = gpar(fill = card_colour, col = border_colour, lwd = 1)
  )
  grid.text(
    "Validation loss", x = 0.12, y = 0.345,
    just = c("left", "centre"),
    gp = gpar(col = text_colour, fontsize = 13, fontfamily = sans_family,
              fontface = "bold")
  )
  grid.text(
    "nats/token  |  lower is better", x = 0.88, y = 0.345,
    just = c("right", "centre"),
    gp = gpar(col = muted_colour, fontsize = 10.5, fontfamily = sans_family)
  )
  
  x_left <- 0.19
  x_right <- 0.875
  y_bottom <- 0.132
  y_top <- 0.296
  
  x_min <- min(history$update)
  x_max <- max(history$update)
  y_min <- min(history$validation)
  y_max <- max(history$validation)
  y_padding <- max(0.10, 0.03 * (y_max - y_min))
  y_min <- max(0, y_min - y_padding)
  y_max <- y_max + y_padding
  if (x_min == x_max) x_max <- x_min + 1
  if (y_min == y_max) y_max <- y_min + 1
  
  x_pos <- function(update) {
    x_left + (update - x_min) / (x_max - x_min) * (x_right - x_left)
  }
  y_pos <- function(loss) {
    y_bottom + (loss - y_min) / (y_max - y_min) * (y_top - y_bottom)
  }
  
  # Horizontal reference lines and the shared y axis.
  ticks <- seq(y_min, y_max, length.out = 3L)
  for (tick in ticks) {
    y <- y_pos(tick)
    grid.lines(
      x = unit(c(x_left, x_right), "npc"),
      y = unit(c(y, y), "npc"),
      gp = gpar(col = border_colour, lwd = 1)
    )
    grid.text(
      sprintf("%.1f", tick), x = x_left - 0.012, y = y,
      just = c("right", "centre"),
      gp = gpar(col = muted_colour, fontsize = 10, fontfamily = sans_family)
    )
  }
  
  grid.text(
    format(x_min, big.mark = ",", trim = TRUE),
    x = x_left, y = 0.101, just = c("left", "centre"),
    gp = gpar(col = muted_colour, fontsize = 10, fontfamily = sans_family)
  )
  grid.text(
    sprintf("Update %s", format(x_max, big.mark = ",", trim = TRUE)),
    x = x_right, y = 0.101, just = c("right", "centre"),
    gp = gpar(col = muted_colour, fontsize = 10, fontfamily = sans_family)
  )
  
  seen <- history[history$update <= current_update, , drop = FALSE]
  if (!nrow(seen)) return(invisible(NULL))
  x <- x_pos(seen$update)
  y <- y_pos(seen$validation)
  if (nrow(seen) >= 2L) {
    grid.lines(
      x = unit(x, "npc"), y = unit(y, "npc"),
      gp = gpar(col = accent_colour, lwd = 2.5, lineend = "round")
    )
  }
  grid.circle(
    x = tail(x, 1L), y = tail(y, 1L), r = unit(4.2, "pt"),
    gp = gpar(fill = accent_colour, col = card_colour, lwd = 1.2)
  )
  invisible(NULL)
}

# Preserve existing line breaks and insert new breaks after at most 30
# characters. Do this to the COMPLETE sample before generating any frames;
# rewrapping partially typed text would make earlier characters jump around.
wrap_body <- function(text, max_chars = max_line_chars) {
  lines <- strsplit(text, "\n", fixed = TRUE)[[1L]]
  if (!length(lines)) return("")
  wrapped <- vapply(lines, function(line) {
    n <- nchar(line, type = "chars")
    if (n <= max_chars) return(line)
    starts <- seq.int(1L, n, by = max_chars)
    paste(substring(line, starts, pmin(starts + max_chars - 1L, n)),
          collapse = "\n")
  }, character(1L))
  paste(wrapped, collapse = "\n")
}

render_frame <- function(info, typed_text, history, cursor_on = TRUE) {
  display_text <- typed_text  # Already wrapped before preview/frame generation.
  if (cursor_on) display_text <- paste0(display_text, "|")
  
  img <- image_graph(width = width, height = height, res = 144, bg = bg_colour)
  grid.newpage()
  grid.rect(gp = gpar(fill = bg_colour, col = NA))
  
  grid.text(
    "GPT-2 training sample progression", x = 0.10, y = 0.957,
    just = c("left", "centre"),
    gp = gpar(col = text_colour, fontsize = 18, fontfamily = sans_family,
              fontface = "bold")
  )
  
  # All metadata fields occupy fixed positions for every checkpoint.
  draw_metric(0.10, 0.23, "UPDATE ", format(info$update, big.mark = ",", trim = TRUE))
  draw_metric(0.345, 0.29, "VAL. LOSS ", sprintf("%.4f", info$validation))
  draw_metric(0.650, 0.125, "TOP-K ", as.character(info$top_k))
  draw_metric(0.780, 0.13, "SEED ", as.character(info$seed))
  
  # Unlabelled message card: remove both the Assistant avatar and its heading.
  grid.roundrect(
    x = 0.50, y = 0.622, width = 0.82, height = 0.431,
    r = unit(10, "pt"),
    gp = gpar(fill = card_colour, col = border_colour, lwd = 1)
  )
  
  grid.text(
    sprintf('Prompt: "%s"    |    temperature %s',
            info$prompt, format(info$temperature, trim = TRUE)),
    x = 0.12, y = 0.787, just = c("left", "centre"),
    gp = gpar(col = muted_colour, fontsize = 12, fontfamily = sans_family)
  )
  grid.lines(
    x = unit(c(0.12, 0.88), "npc"), y = unit(c(0.750, 0.750), "npc"),
    gp = gpar(col = border_colour, lwd = 1)
  )
  grid.text(
    display_text, x = 0.12, y = 0.716, just = c("left", "top"),
    gp = gpar(col = text_colour, fontsize = 13, fontfamily = mono_family)
  )
  
  draw_validation_chart(history, info$update)
  dev.off()
  img
}

# Share typing positions between the preflight frame count and rendering.
reveal_points <- function(text) {
  n <- nchar(text, type = "chars")
  unique(c(seq.int(0L, n, by = chars_per_frame), n))
}

build_frames_for_block <- function(info, history, render) {
  text <- info$body
  positions <- reveal_points(text)
  frames <- list()
  
  for (i in seq_len(hold_start_frames)) {
    frames[[length(frames) + 1L]] <- render(
      info, "", history, cursor_on = (i %% 2L == 1L)
    )
  }
  
  for (i in seq_along(positions)) {
    frames[[length(frames) + 1L]] <- render(
      info, substr(text, 1L, positions[i]), history,
      cursor_on = (i %% 2L == 1L)
    )
  }
  
  for (i in seq_len(hold_end_frames)) {
    frames[[length(frames) + 1L]] <- render(
      info, text, history, cursor_on = FALSE
    )
  }
  
  frames
}

# Read ALL checkpoints before selecting a quick subset for the GIF. The chart
# must keep the entire measured history even when only three messages appear.
raw_text <- read_text_file(input_path)
blocks <- split_blocks(raw_text)
all_samples <- lapply(blocks, parse_block)
all_samples <- all_samples[order(vapply(all_samples, `[[`, integer(1), "update"))]

# Wrap every full result BEFORE the static preview, frame count or GIF begins.
# Only the in-memory display copy changes: samples.txt stays untouched.
for (i in seq_along(all_samples)) {
  all_samples[[i]]$body <- wrap_body(all_samples[[i]]$body)
}

history <- data.frame(
  update = vapply(all_samples, `[[`, integer(1), "update"),
  validation = vapply(all_samples, `[[`, numeric(1), "validation")
)
if (anyDuplicated(history$update)) stop("Duplicate update headers in samples.txt")
if (any(!is.finite(history$validation))) stop("Invalid validation-loss value")

updates <- history$update
preview <- all_samples[[which.min(abs(updates - preview_update))]]
image_write(
  render_frame(preview, preview$body, history, cursor_on = FALSE),
  path = preview_png, format = "png"
)
cat(sprintf("Static preview: %s (update %d)\n", preview_png, preview$update))

if (!preview_only) {
  displayed_samples <- all_samples
  
  if (fast_test && length(all_samples) > 2L) {
    displayed_samples <- all_samples[unique(c(
      1L, as.integer(ceiling(length(all_samples) / 2)), length(all_samples)
    ))]
  }
  
  # Count exactly the frames that will be rendered for the selected samples.
  frames_per_sample <- vapply(
    displayed_samples,
    function(info) {
      hold_start_frames +
        length(reveal_points(info$body)) +
        hold_end_frames +
        block_pause_frames
    },
    integer(1)
  )
  
  total_frames <- sum(frames_per_sample)
  progress <- new.env(parent = emptyenv())
  progress$done <- 0L
  
  cat(sprintf(
    "Preparing %d samples | %d frames | animation duration %.1f seconds\n",
    length(displayed_samples), total_frames, total_frames / fps
  ))
  
  # All animated frames go through this wrapper, including pause frames.
  render_animated <- function(info, typed_text, history, cursor_on = TRUE) {
    frame <- render_frame(info, typed_text, history, cursor_on = cursor_on)
    progress$done <- progress$done + 1L
    
    if (progress$done == 1L ||
        progress$done %% 10L == 0L ||
        progress$done == total_frames) {
      cat(sprintf(
        "\rRendering frames: %d/%d (%.1f%%)",
        progress$done, total_frames, 100 * progress$done / total_frames
      ))
      flush.console()
    }
    
    frame
  }
  
  all_frames <- list()
  
  for (info in displayed_samples) {
    frames <- build_frames_for_block(info, history, render = render_animated)
    all_frames <- c(all_frames, frames)
    
    for (j in seq_len(block_pause_frames)) {
      all_frames[[length(all_frames) + 1L]] <- render_animated(
        info, info$body, history, cursor_on = FALSE
      )
    }
  }
  
  cat("\nAll frames rendered.\n")
  cat("Encoding GIF...\n")
  flush.console()
  
  animation <- image_animate(image_join(all_frames), fps = fps)
  
  cat("Writing GIF...\n")
  flush.console()
  image_write(animation, path = output_gif)
  
  cat("Animated GIF saved: ", output_gif, "\n", sep = "")
}
