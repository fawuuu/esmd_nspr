##### Code for Experiments for Figure 2 #####

source("algorithms.R")

# Define Mirror descent (EG formulation)
# without saving all iterates
# A: measurement matrix
# y: vector of observations
# ini: index for initialization (i_max)
# step: step size
# beta: mirror map parameter
# iteration: maximum number of iterations
# x_star: signal to compute error (without saving iterates)
mirror_eg = function(A, y, ini, step = 0.3, beta = 1e-3,
                    iteration = 100, x_star){
  m = nrow(A)
  n = ncol(A)
  res = rep(0, iteration)

  # Initialization
  size_est = sqrt(mean(y))

  u_cur = rep(beta / 2, n)
  v_cur = rep(beta / 2, n)

  u_cur[ini] = size_est / (2 * sqrt(3)) + sqrt(size_est^2 / 12 + beta^2 / 4)
  v_cur[ini] = -size_est / (2 * sqrt(3)) + sqrt(size_est^2 / 12 + beta^2 / 4)

  x_cur = u_cur - v_cur

  res[1] = min(sqrt(sum((x_cur - x_star)^2)),
              sqrt(sum((x_cur + x_star)^2))) / sqrt(sum(x_star^2)) 

  for(t in 2:iteration){
    # Exponentiated gradient updates
    print(t)
    r = step / size_est^3 * wf_grad(x_cur, A, y)

    u_cur = u_cur * exp(-r)
    v_cur = v_cur * exp(r)
    x_cur = u_cur - v_cur

    # Save new error
    res[t] = min(sqrt(sum((x_cur - x_star)^2)),
                sqrt(sum((x_cur + x_star)^2))) / sqrt(sum(x_star^2))
  }
  return(res)
}

########################################
# Run experiment for Figure 1
set.seed(12345)
n = 50000              # Dimension of signal vector x
k = 10                 # Sparsity level of x
m = 1000               # Number of observations
sigma = 0.5            # Noise to signal ratio
iter = 2000            # Number of iterations

# Generate data
data = gen_data_sparse(m, n, k, 0)
ind_sup = sort(t(data$A^2) %*% data$y, index.return = TRUE)$ix[n]

# Run mirror descent with different values for beta
error1 = mirror_eg(data$A, data$y, ini = ind_sup, step = 0.4,
                  beta = 1e-6, iteration = iter, x_star = data$x)
error2 = mirror_eg(data$A, data$y, ini = ind_sup, step = 0.4,
                  beta = 1e-10, iteration = iter, x_star = data$x)
error3 = mirror_eg(data$A, data$y, ini = ind_sup, step = 0.4,
                  beta = 1e-14, iteration = iter, x_star = data$x)

# Generate noisy data
data = gen_data_sparse(m, n, k, sigma)
ind_sup = sort(t(data$A^2) %*% data$y, index.return = TRUE)$ix[n]

# Run mirror descent with different values for beta
error_noise1 = mirror_eg(data$A, data$y, ini = ind_sup, step = 0.4,
                  beta = 1e-6, iteration = iter, x_star = data$x)
error_noise2 = mirror_eg(data$A, data$y, ini = ind_sup, step = 0.4,
                  beta = 1e-10, iteration = iter, x_star = data$x)
error_noise3 = mirror_eg(data$A, data$y, ini = ind_sup, step = 0.4,
                  beta = 1e-14, iteration = iter, x_star = data$x)


########################################
# Plot Figure 3
pdf(file = "plot_beta.pdf", width = 14.4, height = 4.5)

par(mfrow = c(1, 2), mai = c(0.75, 0.95, 0.1, 0.215),
    bg = "transparent", xpd = TRUE)

# Left plot
plot(log(error1), type = "l", lwd = 2.4, col = "red", ylab = "", xlab = "",
    xaxt = "n", ylim = c(-8.5, 0), las = 1, cex.axis = 1.25, xaxs = "i",
    yaxs = "i")
axis(1, at = 250 * (0:8), label = 250 * (0:8), cex.axis = 1.25)
title(xlab = "Iteration t", cex.lab = 1.55, line = 2.5)
title(ylab = "Relative error (log)", cex.lab = 1.55, line = 3.25)
lines(log(error2), col = "blue", lwd = 2.4)
lines(log(error1), col = "black", lwd = 2.4)
legend("topright", legend = c(expression(paste(beta, "=", 10^{-6})),
                            expression(paste(beta, "=", 10^{-10})),
                            expression(paste(beta, "=", 10^{-14}))),
       col = c("black", "blue", "red"), lwd = 2.4, cex = 1.3,
       inset = c(0.025, 0.025), bty = "n")

# Right plot
plot(log(error_noise3), type = "l", lwd = 2.4, col = "red", ylab = "",
    xlab = "", xaxt = "n", ylim = c(-4,0), las = 1, cex.axis = 1.25, xaxs = "i",
    yaxs = "i")
axis(1, at = 250 * (0:8), label = 250 * (0:8), cex.axis = 1.25)
title(xlab = "Iteration t", cex.lab = 1.55, line = 2.5)
title(ylab = "Relative error (log)", cex.lab = 1.55, line = 3.25)
lines(log(error_noise2), col = "blue", lwd = 2.4)
lines(log(error_noise1), col = "black", lwd = 2.4)
legend("topright", legend = c(expression(paste(beta, "=", 10^{-6})),
                            expression(paste(beta, "=", 10^{-10})),
                            expression(paste(beta, "=", 10^{-14}))),
       col = c("black", "blue", "red"), lwd = 2.4, cex = 1.3, 
       inset = c(0.025, 0.025), bty = "n")

dev.off()