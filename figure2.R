##### Code for Experiments for Figure 2 #####

library(foreach)
library(doParallel)

source("algorithms.R")

# Define wrapper for the experiments on stopping time varying beta
get_time_beta = function(m, n, k, sigma, beta){
  # Generate data
  data = gen_data_sparse(m, n, k, sigma)
  # Estimate coordinate for initialization
  ind_sup = sort(t(data$A^2) %*% data$y, index.return = TRUE)$ix[n]

  time = rep(0, length(beta))
  # Signal reconstruction using mirror descent
  for(i in 1:length(beta)){
    res = mirror_eg(data$A, data$y, beta[i],
                    step = 0.3, ini = ind_sup, iteration = 5000)
    ind = which(data$x != 0)
    for(j in 1:nrow(res)){
      if(min(abs(res[j, ind] / data$x[ind])) > 0.5){
        time[i] = j
        break
      }
    }
  }
  # Return stopping time T_1
  return(time)
}

# Define wrapper for the experiments on stopping time varying sparsity
get_time_sparsity = function(m, n, k, sigma, beta){
  time = rep(0, length(k))
  # Signal reconstruction using mirror descent
  for(i in 1:length(k)){
    # Generate data
    data = gen_data_sparse(m, n, k[i], sigma)
    # Estimate coordinate for initialization
    ind_sup = sort(t(data$A^2) %*% data$y, index.return = TRUE)$ix[n]
    res = mirror_eg(data$A, data$y, beta,
                    step = 0.3, ini = ind_sup, iteration = 5000)
    ind = which(data$x != 0)
    for(j in 1:nrow(res)){
      if(min(abs(res[j, ind] / data$x[ind])) > 0.5){
        time[i] = j
        break
      }
    }
  }
  # Return stopping time T_1
  return(time)
}

########################################
# Experiments on the effect of parameter beta
n = 2000                # Dimension of signal vector x
k = 10                  # Sparsity level of x
m = 1500                # Number of observations
sigma = 0.1             # Noise level
beta = 10^(-4 * (1:10)) # Mirror map parameter
num_rep = 100           # Number of Monte Carlo trials

# Parallelize computation
numcl = 34
cl <- makeCluster(numcl)
registerDoParallel(cl)
clusterExport(cl, list("gen_data_sparse", "mirror_eg",
                      "wf_loss", "wf_grad", "get_time"))
clusterExport(cl, list("n", "k", "m", "sigma", "beta", "num_rep"),
              envir = environment())

# Run mirror descent num_rep times with the specified parameters
time_beta <- foreach(rep = 1:num_rep, .combine = "cbind",
                  .multicombine = TRUE) %dopar% {
  get_time_beta(m, n, k, sigma, beta)
}
stopCluster(cl)

# Save result
saveRDS(time_beta, file = "output_time_beta.rds")

########################################
# Experiments on the effect of sparsity
n = 2000                # Dimension of signal vector x
k = 5 + 2 * (0:10)      # Sparsity level of x
m = 4000                # Number of observations
sigma = 0.1             # Noise level
beta = 10^(-20)         # Mirror map parameter
num_rep = 100           # Number of Monte Carlo trials  

numcl = 34
cl <- makeCluster(numcl)
registerDoParallel(cl)
clusterExport(cl, list("gen_data_sparse", "mirror_eg",
                      "wf_loss", "wf_grad", "get_time", "find_time"))
clusterExport(cl, list("n", "k", "m", "sigma", "beta", "num_rep"),
              envir = environment())

# Run mirror descent num_rep times with the specified parameters
time_k <- foreach(rep = 1:num_rep, .combine = "cbind",
                  .multicombine = TRUE) %dopar% {
  get_time_sparsity(m, n, k, sigma, beta)
}
stopCluster(cl)

# Save result
saveRDS(time_k, file = "output_time_sparsity.rds")


########################################
# Plot Figure 2
pdf(file = "plot_time.pdf", width = 14.4, height = 4.5)

par(mfrow = c(1, 2), mai = c(0.75, 0.95, 0.1, 0.215),
    bg = "transparent", xpd = TRUE)

# Left plot
plot(apply(time_beta, 1, mean), type = "o", lwd = 2.4, cex = 1.2, pch = 16,
    col = "red", ylim = c(0, 2700), ylab = "", xlab = "", xaxt = "n",
    las = 1, cex.axis = 1.25, xaxs = "i", yaxs = "i")
axis(1, at = (1:10), label = 4*(1:10), cex.axis = 1.25)
title(xlab = expression(paste(log[10],
                        "(", "||", x, "*", "||"[2], "/", beta, ")")),
      cex.lab = 1.5, line = 2.7)
title(ylab = expression(paste("Time ", T[1])), cex.lab = 1.55, line = 3.25)
# Add error bars
conf_low = apply(time_beta, 1, mean) - apply(time_beta, 1, sd)
conf_up = apply(time_beta, 1, mean) + apply(time_beta, 1, sd)
polygon(c(1:10, rev(1:10)), c(conf_low, rev(conf_up)),
        col = alpha("red", alpha = 0.3), border = NA)

# Right plot
plot(apply(time_k, 1, mean), type = "o", lwd = 2.4, cex = 1.2, pch = 16,
    col = "red", ylim = c(0, 3000), ylab = "", xlab = "", xaxt = "n",
    las = 1, cex.axis = 1.25, xaxs = "i", yaxs = "i")
axis(1, at = (1:11), label = 5 + 2 * (0:10), cex.axis = 1.25)
title(xlab = "Sparsity level k", cex.lab = 1.55, line = 2.5)
title(ylab = expression(paste("Time ", T[1])), cex.lab = 1.55, line = 3.25)
# Add error bars
conf_low = apply(time_k, 1, mean) - apply(time_k, 1, sd)
conf_up = apply(time_k, 1, mean) + apply(time_k, 1, sd)
polygon(c(1:11, rev(1:11)), c(conf_low, rev(conf_up)),
        col = alpha("red", alpha = 0.3), border = NA)

dev.off()