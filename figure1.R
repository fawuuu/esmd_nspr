##### Code for Experiments for Figure 1 #####

library(foreach)
library(doParallel)
library(abind)
library(scales)

source("algorithms.R")

# Define wrapper for the experiments with oracle and hold-out stopping time
get_result = function(m, n, k, sigma){
  # Generate data
  data = gen_data_sparse(m, n, k, sigma)
  # Estimate coordinate for initialization
  ind_sup = sort(t(data$A^2) %*% data$y, index.return = TRUE)$ix[n]

  # Signal reconstruction using mirror descent + oracle method for stopping time
  res = mirror_eg(data$A, data$y, beta = 1e-20,
                  step = 0.3, ini = ind_sup, iteration = 5000)
  error = rep(0, nrow(res))
  for(j in 1:nrow(res)){
    error[j] = min(sqrt(sum((res[j,] - data$x)^2)), 
                  sqrt(sum((res[j,] + data$x)^2))) / sqrt(sum(data$x)^2)
  }

  # Signal reconstruction using mirror descent
  res = mirror_eg(data$A[1:(0.9 * m),], data$y[1:(0.9 * m)], beta = 1e-20,
                  step = 0.3, ini = ind_sup, iteration = 5000)
  # Selecting stopping time using hold-out method
  error_cv = rep(0, nrow(res))
  error_obj = rep(0, nrow(res))
  for(j in 1:nrow(res)){
    error_cv[j] = min(sqrt(sum((res[j, ] - data$x)^2)),
                      sqrt(sum((res[j, ] + data$x)^2))) / sqrt(sum(data$x^2))
    error_obj[j] = wf_loss(res[j,], data$A[(0.9 * m + 1):m, ],
                           data$y[(0.9 * m + 1):m])
  }
  # Return errors from oracle stopping time and hold-out stopping time
  return(c(min(error) / sqrt(sum(data$x^2)),
          error_cv[which.min(error_obj)] / sqrt(sum(data$x^2)),
          which.min(error), which.min(error_obj)))
}

acomb <- function(...) abind(..., along = 2)

########################################
# Experiments on the effect of noise level
n = 2000               # Dimension of signal vector x
k = 10                 # Sparsity level of x
m = 2000               # Number of observations
sigma = 0.1 * (0:10)   # Noise to signal ratio
num_rep = 100          # Number of Monte Carlo trials

# Parallelize computation
numcl = 34
cl <- makeCluster(numcl)
registerDoParallel(cl)
clusterExport(cl, list("gen_data_sparse", "mirror_eg",
                       "wf_loss", "wf_grad", "get_result"))
clusterExport(cl, list("n", "k", "m", "sigma", "num_rep"),
                       envir = environment())

# Run mirror descent num_rep times with the specified parameters
mmse_noise <- foreach(rep = 1:num_rep, .combine = "acomb",
                .multicombine = TRUE) %dopar% {
  B = array(1, dim = c(length(sigma), 1, 4))
  for(i in 1:length(sigma)){
    B[i,1,] = get_result(m, n, k, sigma[i])
  }
  B
}
stopCluster(cl)

# Save result
saveRDS(mmse_noise, file = "output_noise.rds")

########################################
# Experiments on the effect of number of measurements
n = 2000               # Dimension of signal vector x
k = 10                 # Sparsity level of x
m = 500 * (2:10)       # Number of observations
sigma = 0.1            # Noise to signal ratio
num_rep = 100          # Number of Monte Carlo trials

# Parallelize computation
numcl = 34
cl <- makeCluster(numcl)
registerDoParallel(cl)
clusterExport(cl, list("gen_data_sparse", "mirror_eg",
                       "wf_loss", "wf_grad", "get_result"))
clusterExport(cl, list("n", "k", "m", "sigma", "num_rep"),
                       envir = environment())

# Run mirror descent num_rep times with the specified parameters
mmse_m <- foreach(rep = 1:num_rep, .combine = "acomb",
               .multicombine = TRUE) %dopar% {
  B = array(1, dim = c(length(m), 1, 4))
  for(i in 1:length(m)){
    B[i,1, ] = get_result(m[i], n, k, sigma)
  }
  B
}
stopCluster(cl)

# Save result
saveRDS(mmse_m, file = "output_measurement.rds")

########################################
# Experiments on the effect sparsity
n = 2000               # Dimension of signal vector x
k = 5 + 2 * (0:10)     # Sparsity level of x
m = 4000               # Number of observations
sigma = 0.1            # Noise to signal ratio
num_rep = 100          # Number of Monte Carlo trials

# Parallelize computation
numcl = 34
cl <- makeCluster(numcl)
registerDoParallel(cl)
clusterExport(cl, list("gen_data_sparse", "mirror_eg",
                       "wf_loss", "wf_grad", "get_result"))
clusterExport(cl, list("n", "k", "m", "sigma", "num_rep"),
              envir = environment())

# Run mirror descent num_rep times with the specified parameters
mmse_k <- foreach(rep = 1:num_rep, .combine = "acomb",
                  .multicombine = TRUE) %dopar% {
  B = array(1, dim = c(length(k), 1, 4))
  for(i in 1:length(k)){
    B[i,1, ] = get_result(m, n, k[i], sigma)
  }
  B
}
stopCluster(cl)

# Save result
saveRDS(mmse_k, file = "output_sparsity.rds")


########################################
# Plot Figure 1
pdf(file = "plot_rates.pdf", width = 10.5, height = 3.15)

par(mfrow = c(1, 3), mai = c(0.45, 0.57, 0.1, 0.1),
    mgp = c(3, 0.65, 0), bg = "transparent", xpd = TRUE)

# Left plot
plot(apply(mmse_noise[, , 2], 1, mean), type = "o", lwd = 2, cex = 1, pch = 4,
    col = "blue", ylim = c(0, 0.08), ylab = "", xlab = "", xaxt = "n", las = 1,
    cex.axis = 1.2, xaxs = "i", yaxs = "i")
axis(1, at = (1:11), label = c(0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1),
     cex.axis = 1.2)
title(xlab = expression(paste(sigma, "/||", x, "*", "||"[2]^2)),
     cex.lab = 1.3, line = 2.3)
title(ylab = "Relative error", cex.lab = 1.4)
lines(apply(mmse_noise[, , 1], 1, mean), type = "o", lwd = 2, cex = 1.2,
      pch = 16, col = "red")
# Add error bars
conf_low = apply(mmse_noise[, , 1], 1, mean) - apply(mmse_noise[, , 1], 1, sd)
conf_up = apply(mmse_noise[, , 1], 1, mean) + apply(mmse_noise[, , 1], 1, sd)
conf_low2 = apply(mmse_noise[, , 2], 1, mean) - apply(mmse_noise[, , 2], 1, sd)
conf_up2 = apply(mmse_noise[, , 2], 1, mean) + apply(mmse_noise[, , 2], 1, sd)
polygon(c(1:11, rev(1:11)), c(conf_low2,rev(conf_up2)),
        col = alpha("blue", alpha = 0.3), border = NA)
polygon(c(1:11, rev(1:11)), c(conf_low,rev(conf_up)),
        col = alpha("red", alpha = 0.3), border = NA)

legend("topleft", legend = c("MD", "MD-HO"), col = c("red", "blue"),
       pch = c(16, 4), lwd = 2, cex = 1.2, inset = c(0.025, 0.025), bty = "n")


# Center plot, log-log scale
df = data.frame(X = log(1000 + 500 * (1:8)),
                Y = log(apply(mmse_m[-1,,1], 1, mean)))
df2 = data.frame(X = log(1000 + 500 * (1:8)),
                 Y = log(apply(mmse_m[-1,,2], 1, mean)))

plot(df2, type = "o", lwd = 2, cex = 1, pch = 4, col = "blue",
    ylim = log(c(0.0015, 0.0065)), ylab = "", xlab = "", xaxt = "n",
    yaxt = "n", las = 1, xaxs = "i", yaxs = "i")
axis(1, at = df$X, label = c(1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000),
    cex.axis = 1.2)
axis(2, at = log(c(0.002, 0.003, 0.004, 0.005, 0.006)),
    label = c(0.002, 0.003, 0.004, 0.005, 0.006), cex.axis = 1.2, las = 1)
title(xlab = "Number of measurements m", cex.lab = 1.4, line = 2.1)
title(ylab = "Relative error", cex.lab = 1.4, line = 3.25)
lines(df, type = "o", lwd = 2, cex = 1.2, pch = 16, col = "red")
# Add error bars
conf_low = apply(mmse_m[-1, , 1], 1, mean) - apply(mmse_m[-1, , 1], 1, sd)
conf_up = apply(mmse_m[-1, , 1], 1, mean) + apply(mmse_m[-1, , 1], 1, sd)
conf_low2 = apply(mmse_m[-1, , 2], 1, mean) - apply(mmse_m[-1, , 2], 1, sd)
conf_up2 = apply(mmse_m[-1, , 2], 1, mean) + apply(mmse_m[-1, , 2], 1, sd)
polygon(c(df$X, rev(df$X)), log(c(conf_low2, rev(conf_up2))),
        col = alpha("blue", alpha = 0.3), border = NA) 
polygon(c(df$X, rev(df$X)), log(c(conf_low, rev(conf_up))),
        col = alpha("red", alpha = 0.3), border = NA)

legend("topright", legend = c("MD", "MD-HO"), col = c("red", "blue"),
       pch = c(16, 4), lwd = 2, cex = 1.2, inset = c(0.025, 0.025), bty = "n")

# Right plot, log-log scale
df = data.frame(X = log(5 + 2 * (0:10)),
                Y = log(apply(mmse_k[, , 1], 1, mean)))
df2 = data.frame(X = log(5 + 2 * (0:10)),
                 Y = log(apply(mmse_k[, , 2], 1, mean)))

plot(df2, type = "o", lwd = 2, cex = 1, pch = 4, col = "blue",
     ylim = log(c(0.00097, 0.0051)), ylab = "", xlab = "", xaxt = "n",
     yaxt = "n", las = 1, xaxs = "i", yaxs = "i")
axis(1, at = log(c(5 + 2 * (0:10))), label = c(5 + 2 * (0:10)), cex.axis = 1.2)
axis(2, at = log(c(0.001, 0.002, 0.003, 0.004, 0.005)),
    label = c(0.001, 0.002, 0.003, 0.004, 0.005), cex.axis = 1.2, las = 1)
title(xlab = "Sparsity level k", cex.lab = 1.4, line = 2.1)
title(ylab = "Relative error", cex.lab = 1.4, line = 3.25)
lines(df, type = "o", lwd = 2, cex = 1.2, pch = 16, col = "red")
# Add error bars
conf_low = apply(mmse_k[, , 1], 1, mean) - apply(mmse_k[, , 1], 1, sd)
conf_up = apply(mmse_k[, , 1], 1, mean) + apply(mmse_k[, , 1], 1, sd)
conf_low2 = apply(mmse_k[, , 2], 1, mean) - apply(mmse_k[, , 2], 1, sd)
conf_up2 = apply(mmse_k[, , 2], 1, mean) + apply(mmse_k[, , 2], 1, sd)
polygon(c(df$X, rev(df$X)), log(c(conf_low2, rev(conf_up2))),
        col = alpha("blue", alpha = 0.3), border = NA)
polygon(c(df$X, rev(df$X)), log(c(conf_low, rev(conf_up))),
        col = alpha("red", alpha = 0.3), border = NA)

legend("topleft", legend = c("MD", "MD-HO"), col = c("red", "blue"),
      pch = c(16, 4), lwd = 2, cex = 1.2, inset = c(0.025, 0.025), bty = "n")

dev.off()