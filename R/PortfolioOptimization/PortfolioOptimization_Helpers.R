samplePriceRatios <- function(PriceRatios, N, StartDate, EndDate, auxiliaryColumns){
  timeSubset <- PriceRatios[(Time >= StartDate) & (Time <= EndDate), ]
  tickers <- colnames(timeSubset)[!(colnames(timeSubset) %in% auxiliaryColumns)]
  validTickers <- colMeans(is.na(timeSubset[,.SD,.SDcols = tickers]))==0
  selectedTickers <- sample(validTickers[validTickers],size=N,replace = F)
  out <- timeSubset[,.SD,.SDcols = c(auxiliaryColumns ,names(selectedTickers))]
  out
}


normalizeWeights <- function(x, alpha, nonnegative){
  if(nonnegative==1){
    nnf_softmax(x,dim=1) * alpha
  }else if (nonnegative==0){
    x / torch_sum(torch_abs(x)) * alpha
  }else if (nonnegative==2){
    y <- (nnf_softmax(x,dim=1) - 0.5) * 2
    y / torch_sum(torch_abs(y)) * alpha
  }
}

# x <- torch_tensor(as.matrix(1:10),dtype = torch_float())
# nnf_softmax(x,dim=1)

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
  initialize = function(N, alpha, nonnegative, start=NULL) {
    if(is.null(start)){
      # self$weights = nn_parameter(normalizeWeights(torch_ones(N,1), alpha))
      self$weights = nn_parameter(normalizeWeights(torch_tensor(as.matrix(rnorm(N,1,1)), dtype = torch_float()), alpha, nonnegative))
    }else{
      self$weights = nn_parameter(normalizeWeights(torch_tensor(as.matrix(start), dtype = torch_float()), alpha, nonnegative))
    }
    self$alpha = alpha
    self$nonnegative = nonnegative
  },
  forward = function(x) {
    weights <- normalizeWeights(self$weights, self$alpha, self$nonnegative)
    computeSharpTensor(x,weights)
  }
)


# optimizeSharp <- function(x, alpha, epochs = 1000, start = NULL, silent = TRUE){
#   if(length(dim(x))==3){
#     x <- torch_tensor(x, dtype = torch_float())
#   }else{
#     x <- torch_unsqueeze(torch_tensor(as.matrix(x), dtype = torch_float()),3)
#   }
#   N <- dim(x)[2]
#   model <- weightsModule(N=N, alpha = alpha, start = start)
#   criterion = function(sharp){-torch_mean(sharp)}
#   optimizer = optim_adam(model$parameters, lr = 0.01)
# 
# 
#   for(e in seq_len(epochs)){
#     optimizer$zero_grad()
#     sharp = model(x)
#     loss = criterion(sharp)
#     loss$backward()
#     optimizer$step()
# 
#     if((!silent) & (e %% 100 == 0)){
#       cat(" Epoch:", e,"Loss: ", loss$item(),"\n")
#     }
#   }
# 
#   weights = normalizeWeights(model$weights, model$alpha)
#   sharp = computeSharpTensor(x,weights)
# 
#   list(
#     weights = as.array(weights),
#     sharp = as.array(sharp)
#   )
# }


# xbatch=x[,,bs[[b]]]
# if(length(dim(xbatch))==2){
#   xbatch = xbatch$view(c(dim(xbatch),1))
# }
# x[,,1:1]
# x[,,c(1)]
# xbatch$view(c(10,20,1))

optimizeSharp <- function(x, alpha, nonnegative, epochs = 1000, numStarts = 1, batchSize = Inf, start = NULL, silent = TRUE){
  if(length(dim(x))==3){
    x <- torch_tensor(x, dtype = torch_float())
  }else{
    x <- torch_unsqueeze(torch_tensor(as.matrix(x), dtype = torch_float()),3)
  }
  N <- dim(x)[2]
  R <- dim(x)[3]

  temps <- lapply(1:numStarts, function(s){
    model <- weightsModule(N=N, alpha = alpha, nonnegative = nonnegative, start = start)
    criterion = function(sharp){-torch_mean(sharp)}
    optimizer = optim_adam(model$parameters, lr = 0.01)

    for(e in seq_len(epochs)){
      bs <- sample(seq_len(R),replace = F)
      bs <- split(bs, ceiling(seq_along(bs)/batchSize))
      for(b in seq_along(bs)){
        xbatch <- x[,,bs[[b]]]
        if(length(dim(xbatch))==2){
          xbatch = xbatch$view(c(dim(xbatch),1))
        }
        optimizer$zero_grad()
        sharp = model(xbatch)
        loss = criterion(sharp)
        loss$backward()
        optimizer$step()
      }
      if((!silent) & (e %% 100 == 0)){
        cat("Start:", s," Epoch:", e,"Loss: ", loss$item(),"\n")
      }
    }

    weights = normalizeWeights(model$weights, model$alpha, nonnegative)
    sharp = computeSharpTensor(x,weights)

    list(
      weights = as.array(weights),
      sharp = as.array(sharp)
    )
  })
  s <- which.max(sapply(temps, function(temp) mean(temp$sharp)))
  temp <- temps[[s]]
  temp
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
    parameterCombinations <- expand.grid(parameters,stringsAsFactors = F)
    out <- setNames(lapply(seq_len(nrow(parameterCombinations)), function(i)
      function(...){
        do.call(model ,c(list(...), parameterCombinations[i,,drop=F]))
      }
    ), str_c(modelName, apply(parameterCombinations,1,function(x) str_sub(deparse1(unlist(x)),2))))
  }
  out
}


# 
# parameters <- list(muInfo = c("known"), sigmaInfo = c("knowsdfsdfsdfsdfdfn"), numStarts = c(10,20))
# # parameters <- list(R = 1000, numStarts = c(10,20))
# # parameters <- list(muInfo = "known", sigmaInfo = "known")
# # parameters <- list(muInfo = c("known","knownMean","constant","estimated"),numStarts = c(10,20))
# model <- copulaSim
# 
# modelName <- deparse(substitute(model))
# parameterCombinations <- expand.grid(parameters,stringsAsFactors = F)
# apply(parameterCombinations,1,function(x) str_sub(deparse1(unlist(x),),2))
# 
# 
# 
# 
# deparse1()
# 
# 
# 
# 
# 
# apply(parameterCombinations,1,function(x) {
#   c(unlist(x))
# })






