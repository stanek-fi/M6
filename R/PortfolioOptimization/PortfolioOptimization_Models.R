equalWeights = function(xpast, x, alpha = 0.25){
  N <- dim(x)[2]
  weights = as.matrix(rep(1/N, N) * alpha)
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}

unfeasibleExPostOptimal = function(xpast, x, alpha = 0.25, epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE){
  temp <- optimizeSharp(x, alpha = alpha, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}

laggedExPostOptimal = function(xpast, x, window = 1000, alpha = 0.25, epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE){
  if(nrow(xpast)>window){
    temp <- optimizeSharp(xpast[(nrow(xpast)-window+1):nrow(xpast),], alpha = alpha, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
    weights <- temp$weights
  }else{
    weights <- rep(1/N,N) * alpha
  }
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}

unfeasibleBootstrap = function(xpast, x, alpha = 0.25, R = 100,  epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE){
  xBootstrap <- lapply(1:R, function(r) x[sample(seq_len(nrow(x)),replace = T)])
  xBootstrap <- array(unlist(xBootstrap),dim=c(dim(x),R))
  temp <- optimizeSharp(xBootstrap, alpha = alpha, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}

unfeasibleAnalyticAproximation <- function(xpast,x, alpha=0.25, logTransform = TRUE){
  if(logTransform){
    xtrans <- log(1 + (x-1))
  }else{
    xtrans <- (x-1)
  }
  mu <- apply(xtrans, 2, mean)
  xtransdm <- as.matrix(apply(xtrans, 2, function(col) col - mean(col)))
  Sigma <- t(xtransdm) %*% xtransdm /nrow(xtransdm)
  SigmaInv <- solve(Sigma)
  weights <- SigmaInv %*% mu
  weights <- as.matrix(weights / sum(weights))
  weights <- normalizeWeights(weights,alpha)
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}

unfeasibleCopula = function(xpast, x, alpha = 0.25, R = 100,  epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE, muInfo = "known"){
  TObs <- nrow(x)
  mu <- apply(x, 2, mean)
  xdm <- as.matrix(apply(x, 2, function(col) col - mean(col)))
  Sigma <- t(xdm) %*% xdm/ TObs 
  mu <- switch(muInfo,
         "known" = mu,  
         "mean" = rep(mean(mu),length(mu)),
         "unknown" = rep(1.000787, length(mu))
  )  
  xSim <- simulateCopula(TObs = TObs, R = R, Sigma = Sigma, mu = mu)
  temp <- optimizeSharp(xSim, alpha = alpha, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}
