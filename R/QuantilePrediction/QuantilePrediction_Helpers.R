AugmentStock <- function(Stock, TimeEnd){
  if(max(Stock$index)+1<TimeEnd){
    temp <- seq(max(Stock$index)+1,TimeEnd,by=1)
    # temp <- temp[!(weekdays(temp) %in% c("sobota", "neděle"))]
    temp <- temp[!(weekdays(temp, abbreviate = T) %in% c("so", "ne"))]
    StockAug <- data.table(
      index = temp
    )
    rbind(Stock, StockAug, fill=TRUE)
  }else{
    Stock
  }
}

standartize <- function(x){(x - mean(x,na.rm=T)) / (sd(x, na.rm = T) + 1e-5)}

# computeQuintile <- function(x){ 
#   findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
# } 

computeQuintile <- function(x){ 
  nas <- is.na(x)
  out <- findInterval(rank(x[!nas])/length(x[!nas]),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
  ifelse(nas,NA,out[cumsum(!nas)])
} 

ComputeRPSTensor <- function(y_pred,y){
  temp <- (y_pred$cumsum(2) - y)^2
  mean(temp$sum(2)/5)
}

ComputeRPSTensorVector <- function(y_pred,y){
  temp <- (y_pred$cumsum(2) - y)^2
  temp$sum(2)/5
}


imputeNA <- function(x){
  ifelse(is.na(x)|is.infinite(x),median(x,na.rm = T),x)
}

imputeFeatures <- function(StocksAggr, featureNames = NULL){
  for(featureName in featureNames){
    StocksAggr[[featureName]] <- imputeNA(StocksAggr[[featureName]])
  }
  StocksAggr
}


standartizeFeatures <- function(StocksAggr, featureNames = NULL){
  otherNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Interval", featureNames))]
  StocksAggr[,c(setNames(lapply(otherNames, function(x) get(x)),otherNames), lapply(.SD, function(x) standartize(x))), .(Interval, M6Dataset), .SDcols = featureNames]
}


subsetTensor <- function(x, rows, isSparse = NULL){
  if(is.null(isSparse)){
    isSparse = try({x$is_coalesced()},silent = T)
  }
  if(isSparse == TRUE){
    # print("asdas")
    rowIndices <- as.array(x$indices()[1,]) + 1L
    selection <- rowIndices[rowIndices %in% rows]
    i <- x$indices()[,selection] + 1L
    i[1,] <- seq_along(selection)
    v <- x$values()[selection]
    # out <- torch_sparse_coo_tensor(i, v, c(nrow(x), ncol(x)))$coalesce()
    out <- torch_sparse_coo_tensor(i, v, c(length(selection), ncol(x)))$coalesce()
  }else{
    out <- x[rows,]
  }
  # print("sdfsd")
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
trainModel <- function(model, criterion, train, test = NULL, validation = NULL, epochs = 10, minibatch = Inf, tempFilePath = NULL, patience = 1, printEvery = Inf, lr = 0.001, weight_decay = 0, isSparse = NULL){
  optimizer = optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
  progress <- data.table(
    epoch = seq_len(epochs),
    loss_train = rep(Inf, epochs),
    loss_test = rep(Inf, epochs),
    loss_validation = rep(Inf, epochs)
  )
  if(is.null(isSparse)){
    isSparse <- c(rep(F,2), rep(T,length(train) - 2))
  }
  
  for(e in 1:epochs){
    
    if(is.numeric(minibatch)){
      temp <- sample(seq_len(nrow(train[[2]])))
      mbs <- split(temp, ceiling(seq_along(temp)/minibatch))
    }else{
      mbs <- minibatch()
    }
    
    model$train()
    for(mb in seq_along(mbs)){
      rows <- mbs[[mb]]
      # x_train_mb <- subsetTensor(x_train, rows = rows)
      # y_train_mb <- subsetTensor(y_train, rows = rows)
      # xtype_train_mb <- subsetTensor(xtype_train, rows = rows)
      # train_mb <- lapply(train, function(x) subsetTensor(x, rows = rows))
      train_mb <- lapply(seq_along(train), function(i) subsetTensor(train[[i]], rows = rows, isSparse = isSparse[i]))
      
      optimizer$zero_grad()
      # y_pred_mb = model(x_train_mb, xtype_train_mb)
      y_pred_mb = do.call(model, train_mb[-1])
      # loss = criterion(y_pred_mb, y_train_mb)
      loss = criterion(y_pred_mb, train_mb[[1]])
      loss$backward()
      optimizer$step()
    }
    model$eval()
    
    progress[e, loss_train := as.array(criterion(do.call(model, train[-1]), train[[1]]))]
    if(!is.null(test)){
      progress[e, loss_test := as.array(criterion(do.call(model, test[-1]), test[[1]]))]
    }
    if(!is.null(validation)){
      progress[e, loss_validation := as.array(criterion(do.call(model, validation[-1]), validation[[1]]))]
    }
    if(e %% printEvery == 0){
      print(str_c("Epoch:", e," train: ", round(progress[e, loss_train], 5)," test:", round(progress[e, loss_test], 5), " validation:", round(progress[e, loss_validation], 5), " Time:", Sys.time()))
    }
    
    if(!is.null(test)){
      ebest <- progress[,which.min(loss_test)]
      if((e == ebest) & !is.null(tempFilePath)){
        torch_save(model,file.path(tempFilePath,str_c("temp",".t7")))
      }
      if(e - ebest >= patience){
        progress <- progress[1:e,]
        break()
      }
    }
    
  }
  
  if(!is.null(tempFilePath) & !is.null(test)){
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

  
# constructFFNN = nn_module(
#   initialize = function(inputSize, layerSizes, layerTransforms) {
#     self$layerSizes <- layerSizes
#     self$layerTransforms <- layerTransforms
#     self$layerSizesAll <- c(inputSize, layerSizes)
#     for(i in seq_along(self$layerSizes)){
#       self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
#     }
#   },
#   forward = function(x) {
#     for(i in seq_along(self$layerSizes)){
#       x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
#     }
#     x
#   },
#   fforward = function(x,state){
#     for(i in seq_along(self$layerSizes)){
#       x <- self$layerTransforms[[i]](nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
#     }
#     x
#   }
# )

constructFFNN = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, layerDropouts = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
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


constructFFNNbn = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, layerDropouts = NULL, layerBatchnorms = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    self$layerBatchnorms <- layerBatchnorms
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
    if(!is.null(self$layerBatchnorms)){
      for(i in seq_along(self$layerSizes)){
        if(self$layerBatchnorms[i]==TRUE){
          self[[str_c("layerBatchnorm_",i)]] <- nn_batch_norm1d(self$layerSizesAll[i+1], track_running_stats = F)
        }
      }
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
      if(!is.null(self$layerBatchnorms) && (self$layerBatchnorms[i] == TRUE)){
        x <- self[[str_c("layerBatchnorm_",i)]](x)
      }
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
    }
    x
  }
)


constructSCNN = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, scfun, scsize, sclags, layerDropouts = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    self$scfun <- scfun
    self$scsize <- scsize
    self$sclags <- sclags
    self$sclayer <- nn_linear(self$scsize * self$sclags, self$layerSizesAll[length(self$layerSizesAll)])
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
  },
  forward = function(x,y) {
    for(i in seq_along(self$layerSizes)){
      xout <- self[[str_c("layer_",i)]](x)
      x <- self$layerTransforms[[i]](xout)
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
    }
    ypred <- x
    sc <- self$scfun(ypred,y)
    
    
    # sclagged <- torch_cat(lapply(1:self$sclags, function(lg) {
    #   # torch_cat(list(torch_ones(lg,self$scsize) * sc$mean(),sc[1:(sc$size(1)-lg),]), 1)
    #   torch_cat(list(torch_zeros(lg,self$scsize), sc[1:(sc$size(1)-lg),]), 1)
    # }), 2)
    
    sclagged <- torch_zeros(nrow(x), self$scsize * self$sclags)
    for(i in 1){
      indices <- torch_ones(nrow(x), dtype = torch_bool())
      scsub <- sc[indices,]
      scsublagged <- torch_cat(lapply(1:self$sclags, function(lg) {
        torch_cat(list(torch_zeros(lg,self$scsize), sc[1:(scsub$size(1)-lg),]), 1)
      }), 2)
      sclagged[indices] <- scsublagged
    }
    
    correction <- self$sclayer(sclagged)
    ypredsc <- self$layerTransforms[[length(self$layerSizes)]](xout + correction)
    ypredsc
  }
)

constructSCNN2 = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, scfun, scsize, sclags, layerDropouts = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    self$scfun <- scfun
    self$scsize <- scsize
    self$sclags <- sclags
    # self$sclayer <- nn_linear(self$scsize * self$sclags, self$layerSizesAll[length(self$layerSizesAll)])
    self$scweight <- nn_parameter(torch_zeros(self$layerSizesAll[length(self$layerSizesAll)],self$scsize * self$sclags))
    self$scbn <- nn_batch_norm1d(self$scsize * self$sclags)
    self$sclayer1 <- nn_linear(self$scsize * self$sclags, 5)
    self$sclayer2 <- nn_linear(5, 5)
    self$sclayer3 <- nn_linear(5, self$layerSizesAll[length(self$layerSizesAll)])
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
  },
  forward = function(x,xtype,y) {
    for(i in seq_along(self$layerSizes)){
      xout <- self[[str_c("layer_",i)]](x)
      x <- self$layerTransforms[[i]](xout)
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
    }
    ypred <- x
    sc <- self$scfun(ypred,y)
    
    # sclagged <- torch_cat(lapply(1:self$sclags, function(lg) {
    #   torch_cat(list(torch_ones(lg,self$scsize) * sc$mean(),sc[1:(sc$size(1)-lg),]), 1)
    # }), 2)
    sclagged <- torch_zeros(nrow(x), self$scsize * self$sclags)
    columns <- xtype$indices()[2,]
    uniqueColumns <- unique(as.array(columns))
    for(i in uniqueColumns){
      indices <- columns == i
      scsub <- sc[indices,]
      scsublagged <- torch_cat(lapply(1:self$sclags, function(lg) {
        torch_cat(list(torch_zeros(lg,self$scsize), sc[1:(scsub$size(1)-lg),]), 1)
      }), 2)
      sclagged[indices] <- scsublagged
    }
    
    # correction <- self$sclayer(sclagged) 
    correction <- self$scbn(sclagged)
    # correction <- torch_einsum('nf,of->no', list(correction, self$scweight))
    correction <- nnf_leaky_relu(self$sclayer1(correction))
    correction <- nnf_leaky_relu(self$sclayer2(correction))
    correction <- nnf_leaky_relu(self$sclayer3(correction))
    
    ypredsc <- self$layerTransforms[[length(self$layerSizes)]](xout + correction)
    ypredsc
  }
)


GenStocksAggr <- function(Stocks, IntervalInfos, featureList, M6Datasets, CheckLeakage = T){
  # CheckLeakage <- T
  # s <- 1
  # IntervalInfo <- IntervalInfos[[1]]
  # colnamesStock <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
  StocksAggr <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
    
    if(s %% 1 == 0){
      print(str_c("Stock:", s, " Time:", Sys.time()))
    }
    
    StockAggr <- lapply(IntervalInfos, function(IntervalInfo){
      Stock <- Stocks[[s]]
      Ticker <- names(Stocks)[s]
      # colnames(Stock) <- colnamesStock
      Stock <- AugmentStock(Stock[index>=IntervalInfo$Start & index<=IntervalInfo$End], IntervalInfo$End)
      Stock[,Interval := findInterval(index,IntervalInfo$TimeBreaks,left.open=T)]
      Stock[,Interval := factor(Interval, levels = seq_along(IntervalInfo$IntervalNames), labels = IntervalInfo$IntervalNames)]
      Stock[,Ticker := Ticker]
      StockAggr <- Stock[,computeFeatures(.SD, .BY, featureList),.(Ticker)]
      
      if(CheckLeakage){
        featureNames <- names(StockAggr)[!(names(StockAggr) %in% c("Ticker", "Interval", "Return"))]
        StockCensored <- copy(Stock)
        StockCensored <- StockCensored[index >= IntervalInfo$CheckLeakageStart, (c("Open", "High", "Low", "Close", "Volume", "Adjusted")) := lapply(1:6,function(x) NA)][]
        StockCensoredAggr <- StockCensored[,computeFeatures(.SD, .BY, featureList),.(Ticker)]
        for(featureName in featureNames){
          same <- identical(StockAggr[, get(featureName)], StockCensoredAggr[, get(featureName)])
          if(!same){
            print(str_c("Possible leakage in ", featureName, ", stock = ", Ticker, " (s=",s,")"))
            # stop(str_c("Possible leakage in ", featureName, ", stock = ", Ticker, " (s=",s,")"))
          }
        }
      }
      StockAggr[,Shift := IntervalInfo$Shift]
      StockAggr[,M6Dataset := M6Datasets[s]]
      return(StockAggr)
    })
    return(do.call(rbind,StockAggr))
  }))
  StocksAggr[, ReturnQuintile := computeQuintile(Return), .(Interval, M6Dataset)]
  StocksAggr[, IntervalStart := as.Date(str_sub(Interval,1,10))]
  StocksAggr[, IntervalEnd := as.Date(str_sub(Interval,14,23))]
  return(StocksAggr)
}



# GenIntervalInfos = function(TimeEnd, OmitLast, Shifts){
#   lapply(Shifts, function(Shift) {
#   # TimeStart <- (TimeEnd - Shift) - (7*4) * 691
#   TimeStart <- (TimeEnd - Shift) - (7*4) * 1000
#   TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
#   IntervalStarts <- TimeBreaks[-length(TimeBreaks)]+1
#   IntervalEnds <- TimeBreaks[-1]
#   IntervalNames <- str_c(IntervalStarts, " : " , IntervalEnds)
#   list(
#     Shift = Shift,
#     TimeBreaks = TimeBreaks,
#     IntervalStarts = IntervalStarts,
#     IntervalEnds = IntervalEnds,
#     IntervalNames = IntervalNames,
#     Start = IntervalStarts[1],
#     End = IntervalEnds[(length(IntervalEnds)-OmitLast)],
#     CheckLeakageStart = IntervalStarts[(length(IntervalStarts)-OmitLast)]
#   )
# })
# }

GenIntervalInfos = function(Submission, Shifts = 0, TimeEnd = as.Date("2023-02-05")){
  # as.Date("2023-01-08") + 4*7
  lapply(Shifts, function(Shift) {
    # TimeStart <- (TimeEnd - Shift) - (7*4) * 691
    TimeStart <- (TimeEnd - Shift) - (7*4) * 1000
    TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
    IntervalStarts <- TimeBreaks[-length(TimeBreaks)]+1
    IntervalEnds <- TimeBreaks[-1]
    IntervalNames <- str_c(IntervalStarts, " : " , IntervalEnds)
    list(
      Shift = Shift,
      TimeBreaks = TimeBreaks,
      IntervalStarts = IntervalStarts,
      IntervalEnds = IntervalEnds,
      IntervalNames = IntervalNames,
      Start = IntervalStarts[1],
      End = IntervalEnds[(length(IntervalEnds)-(12-Submission))],
      CheckLeakageStart = IntervalStarts[(length(IntervalStarts)-(12-Submission))]
    )
  })
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



