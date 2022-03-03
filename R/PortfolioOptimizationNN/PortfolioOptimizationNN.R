library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
library(abind)
rm(list=ls())
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
# source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
# source("R/MetaModel/MetaModel.R")
source("R/PortfolioOptimizationNN/PortfolioOptimizationNN_Helpers.R")


Shifts <- c(0,7,14,21)
Submission = 0
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)


# GenerateReturnArray <- F
# if(GenerateReturnArray){
#   Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
#   ReturnArray <- GenReturnArray(Stocks, IntervalInfos)
#   saveRDS(ReturnArray, file.path("Precomputed","ReturnArray.RDS"))
# }else{
#   ReturnArray <- readRDS(file.path("Precomputed","ReturnArray.RDS"))
# }
GenerateReturnArray <- F
if(GenerateReturnArray){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  ReturnArrays <- lapply(1:10, function(i) {
    print(i)
    temp <- StockNames[M6Dataset==i][order(M6Dataset),.(Symbol,M6Dataset)]
    GenReturnArray(Stocks[temp$Symbol], IntervalInfos)
  })
  saveRDS(ReturnArrays, file.path("Precomputed","ReturnArrays.RDS"))
}else{
  ReturnArrays <- readRDS(file.path("Precomputed","ReturnArrays.RDS"))
}

QuantilePredictions <- readRDS(file.path("Precomputed","QuantilePredictions.RDS"))
QuantilePrediction <- QuantilePredictions$meta
# QuantilePrediction <- QuantilePrediction[M6Dataset==1]



QuantilePredictionSplit <- QuantilePrediction[,.(Split = unique(Split)),Interval][order(as.character(Interval))]
TrainRows <- QuantilePredictionSplit[,which(Split=="Train")]
TestRows <- QuantilePredictionSplit[,which(Split=="Test")]
ValidationRows <- QuantilePredictionSplit[,which(Split=="Validation")]


# QuantilePredictionArray <- GenQuantilePredictionArray(QuantilePrediction)
QuantilePredictionArrays <- lapply(sort(unique(QuantilePrediction$M6Dataset)), function(i)  {
  GenQuantilePredictionArray(QuantilePrediction[M6Dataset == i])
})

ReturnArrays <- lapply(seq_along(ReturnArrays), function(i) {
  temp <- ReturnArrays[[i]][dimnames(QuantilePredictionArrays[[i]])[[1]],,]
  validPeriods <- all(dimnames(temp)[[1]] == dimnames(QuantilePredictionArrays[[i]])[[1]])
  validStocks <- all(dimnames(temp)[[2]] == dimnames(QuantilePredictionArrays[[i]])[[2]])
  if(!(validPeriods & validStocks)){
    warning("invalid")
  }
  temp[is.na(temp)]=0
  temp
})


numDatasets <- length(ReturnArrays)
x_train <- vector(mode="list",numDatasets)
x_test <- vector(mode="list",numDatasets)
x_validation <- vector(mode="list",numDatasets)
y_train <- vector(mode="list",numDatasets)
y_test <- vector(mode="list",numDatasets)
y_validation <- vector(mode="list",numDatasets)

for(i in 1:numDatasets){
  x <- torch_tensor(QuantilePredictionArrays[[i]])
  x <- x$nan_to_num(-1)
  y <- torch_tensor(ReturnArrays[[i]])
  y_train[[i]] <- y[TrainRows,,]
  x_train[[i]] <- x[TrainRows,,]
  y_test[[i]] <- y[TestRows,,]
  x_test[[i]] <- x[TestRows,,]
  y_validation[[i]] <- y[ValidationRows,,]
  x_validation[[i]] <- x[ValidationRows,,]
}


x_train <- torch_cat(x_train, 1)
y_train <- torch_cat(y_train, 1)
x_test <- torch_cat(x_test, 1)
y_test <- torch_cat(y_test, 1)
x_validation <- torch_cat(x_validation, 1)
y_validation <- torch_cat(y_validation, 1)


criterion = function(y_pred, y) {
  # RPS <- ComputeSharpTensor(weights = y_pred, y)
  RPS <- ComputeSharpTensor(weights = y_pred, y, eps = 1e-10)
  -RPS$mean()
}

# baseModel ---------------------------------------------------------------

restriction <- (-2):2
constructModel = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, nanReplacement, alpha = 1) {
    self$alpha <- alpha
    self$nanReplacement <- nanReplacement
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    for(i in seq_along(self$layerSizes)){
      # self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
      # self[[str_c("layer_",i)]] <- nn_linearCustom(self$layerSizesAll[i], self$layerSizesAll[i+1], weightInit = 0, biasInit = 0.01)
      # self[[str_c("layer_",i)]] <- nn_linearCustom(self$layerSizesAll[i], self$layerSizesAll[i+1], weightInit = NULL, biasInit = 0.1, initRange = 0.001)
      # self[[str_c("layer_",i)]] <- nn_linearCustom(self$layerSizesAll[i], self$layerSizesAll[i+1], weightInit = NULL, biasInit = 0.01, initRange = 0.001)
      self[[str_c("layer_",i)]] <- nn_linearCustom(self$layerSizesAll[i], self$layerSizesAll[i+1], weightInit = NULL, initRange = NULL, bias = F, const = 0.01)
      # self[[str_c("layer_",i)]] <- nn_linearRestricted(self$layerSizesAll[i], self$layerSizesAll[i+1], restriction = restriction )
    }
  },
  forward = function(x) {
    invalidStocks <- torch_einsum("nkf->nk",list(x[,,1:5] == -1)) > 0
    for(i in seq_along(self$layerSizes)){
      x <- self[[str_c("layer_",i)]](x)
      if (i == length(self$layerSizes)){
        x <- torch_squeeze(x,3)
        x[invalidStocks] = self$nanReplacement
      }
      x <- self$layerTransforms[[i]](x)
    }
    x * self$alpha
  }
)


inputSize <- dim(x)[3]
layerSizes <- c(1)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {
  x
  # nnf_softmax(x,2)
  # x / x$abs()$sum(2)$unsqueeze(2)
}))

# 
# temp <- nn_linearRestricted(5,1,c(2,2,2,2,4))
# input <- x_train[1:20,1,]

model <- constructModel(inputSize, layerSizes, layerTransforms, nanReplacement = 0, alpha = 0.25)
train <- list(y_train, x_train)
test <- list(y_test, x_test)
validation <- list(y_validation, x_validation)

fit <- trainModel(model = model, criterion = criterion, train = train, test = test, validation = validation,  epochs = 2000, minibatch = Inf, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr=0.01)
model <- fit$model
progress <- fit$progress

model$state_dict()

y_pred <- torch_tensor(matrix(0.01 * 0.25,nrow=dim(train[[2]])[[1]], ncol = 100))
RPS <- ComputeSharpTensor(weights = y_pred, train[[1]])
RPS$mean()
y_pred <- model(train[[2]])
y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
RPS <- ComputeSharpTensor(weights = y_pred, train[[1]])
RPS$mean()
# hist(c(as.array(y_pred)), breaks = 50)

y_pred <- torch_tensor(matrix(0.01 * 0.25, nrow=dim(test[[2]])[[1]], ncol = 100))
RPS <- ComputeSharpTensor(weights = y_pred, test[[1]])
RPS$mean()
y_pred <- model(test[[2]])
y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
RPS <- ComputeSharpTensor(weights = y_pred, test[[1]])
RPS$mean()


y_pred <- torch_tensor(matrix(0.01 * 0.25, nrow=dim(validation[[2]])[[1]], ncol = 100))
RPS <- ComputeSharpTensor(weights = y_pred, validation[[1]])
RPS$mean()
y_pred <- model(validation[[2]])
y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
RPS <- ComputeSharpTensor(weights = y_pred, validation[[1]])
RPS$mean()






coefs <- seq(0,2,by=0.1)
trainDiff <- rep(NA,length(coefs))
testDiff <- rep(NA,length(coefs))
validationDiff <- rep(NA,length(coefs))

for(i in seq_along(coefs)){
  print(i)
  
  state <- model$state_dict()
  # state$layer_1.weight <- torch_tensor(matrix(((-2):(2))* coefs[i],nrow=1))
  # state$layer_1.weight <- torch_tensor(matrix(c(-1,1,1,1,2)* coefs[i],nrow=1))
  # state$layer_1.weight <- torch_tensor(matrix(((2):(-2))* coefs[i],nrow=1))
  state$layer_1.bias <- torch_tensor(0.01)
  state$layer_1.weight <- torch_tensor(matrix(c(-.3,.1,.1,.1,.3)* coefs[i],nrow=1))
  # state$layer_1.bias <- torch_tensor(-0.07)
  model$load_state_dict(state)
  
  y_pred <- torch_tensor(matrix(0.01 * 0.25,nrow=dim(train[[2]])[[1]], ncol = 100))
  RPS <- ComputeSharpTensor(weights = y_pred, train[[1]])
  base <- RPS$mean()
  y_pred <- model(train[[2]])
  y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
  RPS <- ComputeSharpTensor(weights = y_pred, train[[1]])
  trainDiff[i] <- as.array(RPS$mean() - base)
  
  y_pred <- torch_tensor(matrix(0.01 * 0.25, nrow=dim(test[[2]])[[1]], ncol = 100))
  RPS <- ComputeSharpTensor(weights = y_pred, test[[1]])
  base <- RPS$mean()
  y_pred <- model(test[[2]])
  y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
  RPS <- ComputeSharpTensor(weights = y_pred, test[[1]])
  testDiff[i] <- as.array(RPS$mean() - base)
  
  y_pred <- torch_tensor(matrix(0.01 * 0.25, nrow=dim(validation[[2]])[[1]], ncol = 100))
  RPS <- ComputeSharpTensor(weights = y_pred, validation[[1]])
  base <- RPS$mean()
  y_pred <- model(validation[[2]])
  y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
  RPS <- ComputeSharpTensor(weights = y_pred, validation[[1]])
  validationDiff[i] <- as.array(RPS$mean() - base)
}


plot(coefs, trainDiff)
plot(coefs, testDiff)
plot(coefs, validationDiff)











# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# model$state_dict()
# 
# 
# 
# y_pred <- apply(as.array(train[[2]]),1:2,function(x) {
#   if(all(x==-1)){
#     return(0)
#   }else{
#     inform <- (x[5] + x[4] * 0.1) / (x[1] + x[2] * 0.1) 
#     # inform <- (x[1] + x[2] * 0.1) / (x[5] + x[4] * 0.1)
#     treshold <- .2
#     rho <- 0.5
#     if(abs(inform - 1) > treshold){
#       return((1-rho) * 0.1 + (+1) * rho * (inform - 1))
#     } else {
#       return(0.1)
#     }
#   }
# })
# y_pred <- torch_tensor(y_pred)
# # y_pred$abs()$sum(2)
# y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
# RPS <- ComputeSharpTensor(weights = y_pred, train[[1]])
# RPS$mean()
# 
# 
# 
# 
# y_pred <- apply(as.array(test[[2]]),1:2,function(x) {
#   if(all(x==-1)){
#     return(0)
#   }else{
#     inform <- (x[5] + x[4] * 0.1) / (x[1] + x[2] * 0.1)
#     # inform <- (x[1] + x[2] * 0.1) / (x[5] + x[4] * 0.1)
#     treshold <- .1
#     rho <- 0.2
#     if(abs(inform - 1) > treshold){
#       return((1-rho) * 0.1 + (+1) * rho * (inform - 1))
#     } else {
#       return(0.1)
#     }
#   }
# })
# y_pred <- torch_tensor(y_pred)
# # y_pred$abs()$sum(2)
# y_pred <- y_pred / y_pred$abs()$sum(2)$unsqueeze(2) * 0.25
# # y_pred$mean()
# RPS <- ComputeSharpTensor(weights = y_pred, test[[1]])
# RPS$mean()
# 
# 
# 
# 
# 
# # train <- list(torch_clone(y_train), torch_clone(x_train))
# # test <- list(torch_clone(y_train), torch_clone(x_train))
# # # train <- list(torch_clone(y_test[68:106,,]), torch_clone(x_test[68:106,,]))
# # # test <- list(torch_clone(y_test[68:106,,]), torch_clone(x_test[68:106,,]))
# # realizedQuintiles <- apply(as.array(train[[1]]$sum(3)),1,function(x) computeQuintile(x))
# # for(k in 1:nrow(realizedQuintiles)){
# #   # print(k)
# #   for(n in 1:ncol(realizedQuintiles)){
# #     # print(n)
# #     temp <- rep(0,5)
# #     temp[realizedQuintiles[k,n]] <- 1
# #     train[[2]][n,k,] <- temp
# #     test[[2]][n,k,] <- temp
# #   }
# # }
# 
# 
# 
# 
# 
# 
# # y_test[67:67,70,]
# # x_test[67:67,70,]
# # nn <- 500
# # torch_zeros(c(nn,100,20))
# # -torch_ones(c(nn,100,5))
# 
# 
# 
# 
# 
# # junk <- x / x$abs()$sum(2)$unsqueeze(2)
# # junk$abs()$sum(2)
# 
# # inputSize <- dim(x)[3]
# # layerSizes <- c(1)
# # layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {
# #   # nnf_softmax(x,2)
# #   # x / x$abs()$sum(2)$unsqueeze(2)
# #   # x
# #   nnf_softmax(torch_squeeze(x,3),2)
# #   # nnf_softmax(torch_squeeze(x,3)$nan_to_num(-Inf),2)
# #   # nnf_softmax(torch_squeeze(x,3)$nan_to_num(0),2)
# # }))
# # model <- constructModel(inputSize, layerSizes, layerTransforms)
# # train <- list(y_train, x_train)
# # test <- list(y_test, x_test)
# # fit <- trainModel(model = model, train, test, criterion, epochs = 100, minibatch = Inf, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
# # model <- fit$model
# # progress <- fit$progress
# # 
# # 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # x_debug <- x_test[67:106,,]
# # x_debug <- x_debug$nan_to_num(-1)
# # y_debug <- y_test[67:106,,]
# # x <- x_debug
# # self <- list()
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # optimizer = optim_adam(model$parameters, lr = 0.01)
# # 
# # model$state_dict()
# # 
# # optimizer$zero_grad()
# # y_pred <- model(x_debug)
# # loss = criterion(y_pred, y_debug)
# # loss
# # loss$backward()
# # optimizer$step()
# # 
# # 
# # model$state_dict()
# # 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # 
# # if(T){
# #   start <- Sys.time()
# #  
# #   Sys.time() - start 
# #   baseModel <- fit$model
# #   baseModelProgress <- fit$progress
# #   saveRDS(baseModelProgress, file.path("Precomputed","baseModelProgress.RDS"))
# #   torch_save(baseModel, file.path("Precomputed", str_c("baseModel", ".t7")))
# # }else{
# #   baseModelProgress <- readRDS(file.path("Precomputed","baseModelProgress.RDS"))
# #   baseModel <- torch_load(file.path("Precomputed", str_c("baseModel", ".t7")))
# # }
# # 
# # 
# # 
# # inputSize <- dim(x)[3]
# # layerSizes <- c(10, 1)
# # layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {
# #   nnf_softmax(torch_squeeze(x,3)$nan_to_num(-Inf),2)
# # }))
# # baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms)
# # 
# # weights <- baseModel(x_train)
# # criterion(weights, y_train)
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # out <- computeSharpTensor(weights, y_train)$mean()
# # out$mean()
# # 
# # 
# # 
# # 
# # 
# # weights <- baseModel(x[418:419,,])
# # # dim(weights)
# # # sum(weights[2,])
# # # 
# # ysub <- y[418:419,,]
# # 
# # 
# # 
# # # rows <- 1668:1669
# # # rows <- 1500:1501
# # rows <- 9:10
# # weights <- baseModel(x[rows,,])
# # y <- y[rows,,]
# # computeSharpTensor(y,weights)
# # 
# # weights <- baseModel(x)
# # out <- computeSharpTensor(y,weights)
# # which(is.na(as.array(out)))
# # 
# # # weights$sum(2)
# # as.array(weights)
# # as.array(x[rows,,])
# # 
# # 
# # temp <- matrix(c(1:5,rep(1,5)),nrow=5) + 0.01
# # temp[5,] <- NA
# # temp <- torch_tensor(temp)
# # temp <- temp$nan_to_num(-Inf)
# # nnf_softmax(temp,1)
# # 
# # 
# # ret - sret/20
# # 
# # sret[]
# # torch_sub(ret,sret[,])
# # 
# # 
# # sdp <- torch_std(ret,dim=2, unbiased=TRUE)
# # 
# # 
# # 
# # sum(ysub[2,,7] * weights[2,])
# # 
# # 
# # 
# # computeSharpTensor <- function(x, weights) {
# #   RET <- torch_squeeze(torch_einsum("mnp,nk->mkp",list(x-1,weights)),dim=2)
# #   ret <- torch_log(1 + RET)
# #   sret <- torch_einsum("mn->n",ret)
# #   Tobs <- ret$size()[1]
# #   sdp <- torch_std(ret,dim=1, unbiased=TRUE)
# #   ((21*12) / sqrt(252)) * (1/Tobs) * sret/sdp
# # }
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # temp <- as.array(nnf_softmax(baseModel(x[1668:1669,,]),2))
# # 
# # 
# # 
# # 
# # y[1668:1669,]
# # 
# # 
# # all(dimnames(QuantilePredictionArray)[[1]] %in% dimnames(ReturnArray)[[1]])
# # 
# # 
# # x <- torch_tensor(QuantilePredictionArray)
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # # plot(Stocks[["VXX"]]$VXX.Adjusted)
# # 
# # QuantilePrediction <- QuantilePredictions$meta
# # 
# # # temp <- QuantilePrediction[Split == "Test"][order(Rank5 - Rank1)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# # # View(temp)
# # # temp <- QuantilePrediction[Split == "Test"][order(Rank5 + Rank4 - Rank1 - Rank2)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# # # View(temp)
# # submission <- QuantilePrediction[Split=="Validation"]
# # submission <- submission[match(template$ID, submission$Ticker), .(ID = Ticker, Rank1 , Rank2, Rank3, Rank4, Rank5, Decision = 0)]
# # # View(submission[order(Rank5 - Rank1)])
# # # View(submission[order(Rank5 + Rank4 - Rank1 - Rank2)])
# # submission[ID == "VXX", Decision := -.125]
# # submission[ID == "V", Decision := .0625]
# # submission[ID == "CDW", Decision := .0625]
# # 
# # 
# # submission <- validateSubmission(submission, Round = T)
# # 
# # write.csv(submission,file.path("Results","SubmissionManual.csv"),row.names = F, quote=FALSE)
# # 
# # 
# # 
# # 
