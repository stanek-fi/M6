equalWeights = function(xpast, x, alpha = 0.25){
  temp <- optimizeSharp(x, alpha = alpha, epochs = 0)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}

unfeasibleExPostOptimal = function(xpast, x, alpha = 0.25, epochs = 100){
  temp <- optimizeSharp(x, alpha = alpha, epochs = epochs)
  list(
    weights = temp$weights,
    sharp = computeSharpMatrix(x,temp$weights)
  )
}

laggedExPostOptimal = function(xpast, x, window = 1000, alpha = 0.25, epochs = 100){
  if(nrow(xpast)>window){
    temp <- optimizeSharp(xpast[(nrow(xpast)-window+1):nrow(xpast),], alpha = alpha, epochs = epochs)
    weights <- temp$weights
  }else{
    weights <- rep(1/N,N) * alpha
  }
  list(
    weights = weights,
    sharp = computeSharpMatrix(x,weights)
  )
}


