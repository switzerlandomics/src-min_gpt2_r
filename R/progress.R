# Small experiment-log and file-writing helpers. No model or plotting code.

format_duration <- function(seconds) {
  if (length(seconds) != 1L || !is.finite(seconds) || seconds < 0) return("calculating")
  seconds <- as.integer(round(seconds))
  hours <- seconds %/% 3600L
  minutes <- (seconds %% 3600L) %/% 60L
  remaining <- seconds %% 60L
  if (hours > 0L) sprintf("%dh %02dm %02ds", hours, minutes, remaining)
  else if (minutes > 0L) sprintf("%dm %02ds", minutes, remaining)
  else sprintf("%ds", remaining)
}

make_logger <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  function(level, template, ...) {
    line <- sprintf("%s | %-5s | %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                    level, sprintf(template, ...))
    cat(line, "\n", sep = "")
    cat(line, "\n", sep = "", file = path, append = TRUE)
    invisible(line)
  }
}

# Rate/ETA are measured only from updates performed in this session, so
# restarting a run does not distort the remaining-time estimate.
progress_status <- function(completed, total, elapsed,
                            active_completed, active_elapsed, bar_width = 20L) {
  fraction <- if (total > 0) min(1, max(0, completed / total)) else 1
  filled <- as.integer(floor(bar_width * fraction))
  bar <- paste0("[", strrep("=", filled), strrep(".", bar_width - filled), "]")
  rate <- if (active_elapsed > 0 && active_completed > 0)
    active_completed / active_elapsed else NA_real_
  eta <- if (is.finite(rate) && rate > 0)
    max(0, total - completed) / rate else NA_real_
  sprintf("%s %5.1f%% | %d/%d | elapsed %s | ETA %s", bar, 100 * fraction,
          completed, total, format_duration(elapsed), format_duration(eta))
}

# Write in the destination directory and replace only once the new file is
# complete. A copy fallback accommodates systems that cannot rename over an
# existing file; on those systems the fallback is not an atomic replacement.
atomic_write <- function(path, writer) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".writing-", tmpdir = dirname(path))
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  writer(temporary)
  if (!file.rename(temporary, path) &&
      !file.copy(temporary, path, overwrite = TRUE)) {
    stop("Unable to save file: ", path, call. = FALSE)
  }
  invisible(path)
}

write_rds <- function(object, path) {
  atomic_write(path, function(temporary) saveRDS(object, temporary))
}

write_csv <- function(data, path) {
  atomic_write(path, function(temporary) utils::write.csv(data, temporary, row.names = FALSE))
}
