##### Code for "Nearly Minimax-Optimal Rates of Convergence for Noisy
##### Sparse Phase Retrieval via Early-Stopped Mirror Descent"

# Define the function to generate data
# m: number of observations
# n: dimension of the signal x^*
# k: sparsity level
# sigma: noise level
gen_data_sparse = function(m, n, k, sigma = 0){
  # Generate a k-sparse vector x
  x = runif(n, 0.15, 1) * sample(c(-1, 1), n, rep = T)
  ind = sample(1:n, k)
  x[-ind] = 0

  # Generate Gaussian measurement matrix A
  A = matrix(rnorm(m * n), nrow = m, ncol = n)

  # Generate vector of observations y
  y = as.vector((A %*% x)^2) + rnorm(m, sd = sigma * sum(x^2))

  return(list(A = A, x = x, y = y))
}

# Define the empirical risk, squared-magnitude-based loss
wf_loss = function(x, A, y){
  m = length(y)
  return(sum((y - (A %*% x)^2)^2) / (4 * m))
}

# Define the gradient, squared-magnitude-based loss
wf_grad = function(x, A, y){
  m = nrow(A)
  est = A %*% x
  return(t(A) %*% (((est)^2 - y) * est) / m)
}

# Define Mirror descent (EG formulation)
# A: measurement matrix
# y: vector of observations
# ini: index for initialization (i_max)
# step: step size
# beta: mirror map parameter
# iteration: maximum number of iterations
mirror_eg = function(A, y, ini, step = 0.4, beta = 1e-20, iteration = 5000){
  m = nrow(A)
  n = ncol(A)
  X = matrix(0, nrow = iteration, ncol = n)

  # Initialization
  size_est = sqrt(mean(y))

  u_cur = rep(beta / 2, n)
  v_cur = rep(beta / 2, n)

  u_cur[ini] = size_est / (2 * sqrt(3)) + sqrt(size_est^2 / 12 + beta^2 / 4)
  v_cur[ini] = -size_est / (2 * sqrt(3)) + sqrt(size_est^2 / 12 + beta^2 / 4)

  x_cur = u_cur - v_cur
  X[1,] = x_cur

  for(t in 2:iteration){
    # Exponentiated gradient updates
    r = step / size_est^3 * wf_grad(x_cur, A, y)

    u_cur = u_cur * exp(-r)
    v_cur = v_cur * exp(r)
    x_cur = u_cur - v_cur

    # Save new estimate
    X[t,] = x_cur
  }
  return(X)
}