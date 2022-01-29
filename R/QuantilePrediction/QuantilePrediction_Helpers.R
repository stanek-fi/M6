standartize <- function(x){(x - mean(x)) / (sd(x) + 1e-5)}

computeQuintile <- function(x){
  findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
} 

ComputeRPSTensor <- function(y_pred,y){
  temp <- (y_pred$cumsum(2) - y)^2
  mean(temp$sum(2)/5)
}

imputeNA <- function(x){
  ifelse(is.na(x),median(x,na.rm = T),x)
}

imputeFeatures <- function(StocksAggr, featureNames = NULL){
  for(featureName in featureNames){
    StocksAggr[[featureName]] <- imputeNA(StocksAggr[[featureName]])
  }
  StocksAggr
}


standartizeFeatures <- function(StocksAggr, featureNames = NULL){
  otherNames <- names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]
  StocksAggr[,c(setNames(lapply(otherNames, function(x) get(x)),otherNames), lapply(.SD, function(x) standartize(x))), Interval, .SDcols = featureNames]
}


subsetTensor <- function(x, rows, isSparse = NULL){
  if(is.null(isSparse)){
    isSparse = try({x$is_coalesced()},silent = T)
  }
  if(isSparse == TRUE){
    rowIndices <- as.array(x$indices()[1,]) + 1L
    selection <- rowIndices[rowIndices %in% rows]
    i <- x$indices()[,selection] + 1L
    v <- x$values()[selection]
    out <- torch_sparse_coo_tensor(i, v, c(nrow(x), ncol(x)))$coalesce()
  }else{
    out <- x[rows,]
  }
  return(out)
}



minibatchSampler = function(batchSize, xtype_train){
  rows <- as.array(xtype_train$indices()[1,]) + 1
  columns <- as.array(xtype_train$indices()[2,]) + 1
  uniqueColumns <- unique(columns)
  bs <- sample(seq_along(uniqueColumns),replace = F)
  bs <- split(bs, ceiling(seq_along(bs)/batchSize))
  bs <- lapply(bs, function(x) uniqueColumns[x])
  bs <- lapply(bs, function(x) which(columns %in% x))
  bs
}


# optimizeModel <- function(model, y_train, x_train, xtype_train, y_test, x_test, xtype_test, criterion, epochs = 10, minibatch = Inf, tempFilePath = NULL, patience = 1, printEvery = Inf){
trainModel <- function(model, train, test, criterion, epochs = 10, minibatch = Inf, tempFilePath = NULL, patience = 1, printEvery = Inf){
  optimizer = optim_adam(model$parameters, lr = 0.01)
  progress <- data.table(
    epoch = seq_len(epochs),
    loss_train = as.numeric(rep(NA, epochs)),
    loss_test = as.numeric(rep(NA, epochs))
  )
  
  for(e in 1:epochs){
    
    if(is.numeric(minibatch)){
      temp <- sample(seq_len(nrow(train[[2]])))
      mbs <- split(temp, ceiling(seq_along(temp)/minibatch))
    }else{
      mbs <- minibatch()
    }
    
    for(mb in seq_along(mbs)){
      rows <- mbs[[mb]]
      # x_train_mb <- subsetTensor(x_train, rows = rows)
      # y_train_mb <- subsetTensor(y_train, rows = rows)
      # xtype_train_mb <- subsetTensor(xtype_train, rows = rows)
      # train_mb <- lapply(train, function(x) subsetTensor(x, rows = rows))
      isSparse <- c(rep(F,2), rep(T,length(train) - 2))
      train_mb <- lapply(seq_along(train), function(i) subsetTensor(train[[i]], rows = rows, isSparse = isSparse[i]))
      
      optimizer$zero_grad()
      # y_pred_mb = model(x_train_mb, xtype_train_mb)
      y_pred_mb = do.call(model, train_mb[-1])
      # loss = criterion(y_pred_mb, y_train_mb)
      loss = criterion(y_pred_mb, train_mb[[1]])
      loss$backward()
      optimizer$step()
    }
    
    # loss_train_e <- as.array(criterion(model(x_train,xtype_train), y_train))
    loss_train_e <- as.array(criterion(do.call(model, train[-1]), train[[1]]))
    # loss_test_e <- as.array(criterion(model(x_test,xtype_test), y_test))
    loss_test_e <- as.array(criterion(do.call(model, test[-1]), test[[1]]))
    progress[e, loss_train := loss_train_e]
    progress[e, loss_test := loss_test_e]
    if(e %% printEvery == 0){
      print(str_c("Epoch:", e," loss_train: ", round(progress[e, loss_train], 3)," loss_test:", round(progress[e, loss_test], 3), " Time:", Sys.time()))
    }
    
    ebest <- progress[,which.min(loss_test)]
    
    if((e == ebest) & !is.null(tempFilePath)){
      torch_save(model,file.path(tempFilePath,str_c("temp",".t7")))
    }
    
    if(e - ebest >= patience){
      progress <- progress[1:e,]
      break()
    }
  }
  
  if(!is.null(tempFilePath)){
    model <- torch_load(file.path(tempFilePath,str_c("temp",".t7")))
    file.remove(file.path(tempFilePath,str_c("temp",".t7")))
  }
  
  return(list(
    model = model,
    progress = progress
  ))
}


# FFNNSoftmax = nn_module(
#   initialize = function(layerSizes) {
#     self$layerSizes <- layerSizes
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       self[[str_c("layer_",i)]] <- nn_linear(layerSizes[i], layerSizes[i+1])
#     }
#   },
#   forward = function(x) {
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- nnf_relu(self[[str_c("layer_",i)]](x))
#     }
#     nnf_softmax(x,2)
#   },
#   fforward = function(x,state){
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- nnf_relu(nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
#     }
#     nnf_softmax(x,2)
#   }
# )

  
constructFFNN = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
    }
    x
  },
  fforward = function(x,state){
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
    }
    x
  }
)


# FFNNSoftmax = nn_module(
#   initialize = function(layerSizes) {
#     self$layerSizes <- layerSizes
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       self[[str_c("layer_",i)]] <- nn_linear(layerSizes[i], layerSizes[i+1])
#     }
#   },
#   forward = function(x) {
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- self[[str_c("layer_",i)]](x)
#     }
#     nnf_softmax(x,2)
#   },
#   fforward = function(x,state){
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]])
#     }
#     nnf_softmax(x,2)
#   }
# )


# train = list(y_train, x_train, xtype_train)
# test = list(y_test, x_test, xtype_test)
# trainModel <- function(model, train, test, minibatchSampler = NULL, lr = 0.01, epochs = 100){
#   optimizer = optim_adam(metaModel$parameters, lr = lr)
#   progress <- data.table(
#     epoch = seq_len(epochs),
#     loss_train = as.numeric(rep(NA, epochs)),
#     loss_test = as.numeric(rep(NA, epochs))
#   )
#   for(i in 1:epochs){
#     optimizer$zero_grad()
#     y_pred = metaModel(x_train,xtype_train)
#     loss = criterion(y_pred, y_train)
#     loss$backward()
#     optimizer$step()
#     
#     progress[i, loss_train := loss$item()]
#     progress[i, loss_test := as.array(criterion(metaModel(x_test,xtype_test), y_test))]
#     if(i %% 10 == 0){
#       print(str_c("Epoch:", i," loss_train: ", round(progress[i,loss_train],3)," loss_test:", round(progress[i,loss_test],3), " Time:", Sys.time()))
#     }
#   }
# }



