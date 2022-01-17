equalWeights = function(xpast, x, alpha = 0.25){
  N <- dim(x)[2]
  weights = as.matrix(rep(1/N, N) * alpha)
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}

unfeasibleExPostOptimal = function(xpast, x, alpha = 0.25, nonnegative = F, epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE){
  temp <- optimizeSharp(x, alpha = alpha, nonnegative = nonnegative, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}

laggedExPostOptimal = function(xpast, x, window = 1000, alpha = 0.25, nonnegative=F, epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE){
  if(nrow(xpast)>window){
    temp <- optimizeSharp(xpast[(nrow(xpast)-window+1):nrow(xpast),], alpha = alpha, nonnegative = nonnegative, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
    weights <- temp$weights
  }else{
    weights <- rep(1/N,N) * alpha
  }
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}

unfeasibleBootstrap = function(xpast, x, alpha = 0.25, nonnegative = F, R = 100,  epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE){
  xBootstrap <- lapply(1:R, function(r) x[sample(seq_len(nrow(x)),replace = T)])
  xBootstrap <- array(unlist(xBootstrap),dim=c(dim(x),R))
  temp <- optimizeSharp(xBootstrap, alpha = alpha, nonnegative = nonnegative, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}

unfeasibleAnalyticAproximation <- function(xpast,x, nonnegative = F, alpha=0.25, logTransform = TRUE){
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
  weights <- normalizeWeights(weights,alpha,nonnegative)
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}

# unfeasibleCopula = function(xpast, x, alpha = 0.25, R = 100,  epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE, muInfo = "known"){
#   TObs <- nrow(x)
#   mu <- apply(x, 2, mean)
#   xdm <- as.matrix(apply(x, 2, function(col) col - mean(col)))
#   Sigma <- t(xdm) %*% xdm/ TObs 
#   mu <- switch(muInfo,
#          "known" = apply(x, 2, mean),  
#          "mean" = rep(mean(apply(x, 2, mean)),TObs),
#          "unknown" = rep(1.000787, TObs)
#   )  
#   xSim <- simulateCopula(TObs = TObs, R = R, sigma = Sigma, mu = mu)
#   temp <- optimizeSharp(xSim, alpha = alpha, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
#   list(
#     weights = temp$weights,
#     sharp = computeSharpMatrix(x,temp$weights)
#   )
# }

copulaSim = function(xpast, x, alpha = 0.25, nonnegative = F, R = 100,  epochs = 100, numStarts = 1, batchSize = Inf, silent = TRUE, muInfo = "known",  sigmaInfo = "known", lambda = 0.94){
  TObs <- nrow(x)
  N <- ncol(x)
  timeDecay <- sapply(seq_len(nrow(xpast)), function(i) lambda^i)
  timeDecay <- timeDecay /sum(timeDecay)
  sigma <- switch(sigmaInfo,
                  "known" = {
                    xdm <- as.matrix(apply(x, 2, function(col) col - mean(col)))
                    t(xdm) %*% xdm/ TObs 
                  },
                  "estimated" = {
                    xpastdm <- as.matrix(apply(xpast, 2, function(col) (col - sum(col*timeDecay))*sqrt(timeDecay) ))
                    t(xpastdm) %*% xpastdm
                  }
  )
  mu <- switch(muInfo,
               "known" = {
                 apply(x, 2, mean)
               },  
               "knownMean" = {
                 rep(mean(apply(x, 2, mean)),N)
               },
               "constant" = {
                 rep(1.000787, N)
               },
               "estimated" = {
                 as.vector(t(xpast) %*% timeDecay)
               }
  )  
  xSim <- simulateCopula(TObs = TObs, R = R, sigma = sigma, muMarginal = mu)
  temp <- optimizeSharp(xSim, alpha = alpha, nonnegative = nonnegative, epochs = epochs, numStarts = numStarts, batchSize = batchSize, silent = silent)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}



