samplePriceRatios <- function(PriceRatios, N, StartDate, EndDate, auxiliaryColumns){
  timeSubset <- PriceRatios[(Time >= StartDate) & (Time <= EndDate), ]
  tickers <- colnames(timeSubset)[!(colnames(timeSubset) %in% auxiliaryColumns)]
  validTickers <- colMeans(is.na(timeSubset[,.SD,.SDcols = tickers]))==0
  selectedTickers <- sample(validTickers[validTickers],size=N,replace = F)
  out <- timeSubset[,.SD,.SDcols = c(auxiliaryColumns ,names(selectedTickers))]
  out
}


normalizeWeights <- function(x, alpha){
  x / torch_sum(torch_abs(x)) * alpha
}


# computeSharpTensor <- function(x, weights) {
#   RET <- torch_matmul(x - 1, weights)
#   ret <- torch_log(1 + RET)
#   sret <- torch_sum(ret)
#   Tobs <- ret$size()[1]
#   sdp <- torch_sqrt(1/(Tobs - 1) * torch_sum((ret - sret/Tobs)^2))
#   sret / sdp
# }

computeSharpTensor <- function(x, weights) {
  RET <- torch_squeeze(torch_einsum("mnp,nk->mkp",list(x-1,weights)),dim=2)
  ret <- torch_log(1 + RET)
  sret <- torch_einsum("mn->n",ret)
  Tobs <- ret$size()[1]
  sdp <- torch_std(ret,dim=1, unbiased=TRUE)
  ((21*12) / sqrt(252)) * (1/Tobs) * sret/sdp
}

computeSharpMatrix <- function(x, weights) {
  RET <- as.matrix((x - 1)) %*% as.matrix(weights)
  ret <- log(1 + RET)
  sret <- sum(ret)
  Tobs <- nrow(ret)
  sdp <- (1/(Tobs - 1) * sum((ret - sret/Tobs)^2))^(0.5)
  ((21*12) / sqrt(252)) * (1/Tobs) * sret / sdp
}

weightsModule = nn_module(
  initialize = function(N,alpha,start=NULL) {
    if(is.null(start)){
      self$weights = nn_parameter(normalizeWeights(torch_ones(N,1), alpha))
    }else{
      self$weights = nn_parameter(normalizeWeights(torch_tensor(as.matrix(start)), alpha))
    }
    self$alpha = alpha
  },
  forward = function(x) {
    weights <- normalizeWeights(self$weights, self$alpha)
    computeSharpTensor(x,weights)
  }
)


optimizeSharp <- function(x, alpha, epochs = 1000, start = NULL, silent = TRUE){
  # x <- torch_tensor(as.matrix(x), dtype = torch_float())
  if(length(dim(x))==3){
    x <- torch_tensor(as.matrix(x), dtype = torch_float())
  }else{
    # x <- torch_tensor(array(x,dim=c(dim(x),1)), dtype = torch_float())
    x <- torch_unsqueeze(torch_tensor(as.matrix(x), dtype = torch_float()),3)
  }
  N <- dim(x)[2]
  model <- weightsModule(N=N, alpha = alpha, start = start)
  # criterion = function(sharp){-sharp}
  criterion = function(sharp){-torch_mean(sharp)}
  optimizer = optim_adam(model$parameters, lr = 0.01)
  
  for(i in seq_len(epochs)){
    optimizer$zero_grad()
    sharp = model(x)
    loss = criterion(sharp)
    loss$backward()
    optimizer$step()
    
    if((!silent) & (i %% 100 == 0)){
      cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
    }
  }
  
  weights = normalizeWeights(model$weights, model$alpha)
  sharp = computeSharpTensor(x,weights)
  
  list(
    weights = as.array(weights),
    sharp = as.array(sharp)
  )
}

createModelCombinations <- function(model,parameters=NULL){
  modelName <- deparse(substitute(model))
  if(is.null(parameters)){
    out <- setNames(list(
      function(...){
        model(...)
      }
    ),modelName)
  }else{
    parameterCombinations <- expand.grid(parameters)
    out <- setNames(lapply(seq_len(nrow(parameterCombinations)), function(i)
      function(...){
        do.call(model ,c(list(...), parameterCombinations[i,,drop=F]))
      }
    ), str_c(modelName, apply(parameterCombinations,1,function(x) str_sub(deparse(unlist(x)),2))))
  }
  out
}



