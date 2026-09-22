# Manual Adam; parameters and gradients have the same nested-list structure.
zeros_like <- function(values) {
  if (is.list(values)) return(lapply(values, zeros_like))
  values * 0
}

adam_initialise <- function(parameters) {
  list(moment1 = zeros_like(parameters), moment2 = zeros_like(parameters), step = 0L)
}

adam_update <- function(parameters, gradients, state, learning_rate = 0.001,
                        beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
                        clip_norm = 1) {
  if (!is.finite(learning_rate) || learning_rate <= 0) stop("Invalid learning_rate.")
  norm <- sqrt(sum(unlist(gradients, use.names = FALSE)^2))
  if (!is.finite(norm)) stop("Non-finite gradients; update aborted.")
  if (is.finite(clip_norm) && clip_norm > 0 && norm > clip_norm) {
    scale <- clip_norm / norm
  } else {
    scale <- 1
  }
  step <- state$step + 1L
  walk <- function(p, g, m, v) {
    if (is.list(p)) {
      items <- lapply(seq_along(p), function(i) walk(p[[i]], g[[i]], m[[i]], v[[i]]))
      return(list(parameters = setNames(lapply(items, `[[`, "parameters"), names(p)),
                  moment1 = setNames(lapply(items, `[[`, "moment1"), names(p)),
                  moment2 = setNames(lapply(items, `[[`, "moment2"), names(p))))
    }
    g <- g * scale
    m <- beta1 * m + (1 - beta1) * g
    v <- beta2 * v + (1 - beta2) * g^2
    m_hat <- m / (1 - beta1^step)
    v_hat <- v / (1 - beta2^step)
    list(parameters = p - learning_rate * m_hat / (sqrt(v_hat) + epsilon),
         moment1 = m, moment2 = v)
  }
  updated <- walk(parameters, gradients, state$moment1, state$moment2)
  list(parameters = updated$parameters,
       state = list(moment1 = updated$moment1, moment2 = updated$moment2, step = step),
       gradient_norm = norm, gradient_scale = scale)
}
