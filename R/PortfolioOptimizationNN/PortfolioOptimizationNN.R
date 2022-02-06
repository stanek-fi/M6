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


GenerateReturnArray <- F
if(GenerateReturnArray){
  Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
  ReturnArray <- GenReturnArray(Stocks, IntervalInfos)
  saveRDS(ReturnArray, file.path("Precomputed","ReturnArray.RDS"))
}else{
  ReturnArray <- readRDS(file.path("Precomputed","ReturnArray.RDS"))
}

QuantilePredictions <- readRDS(file.path("Precomputed","QuantilePredictions.RDS"))
QuantilePrediction <- QuantilePredictions$meta
QuantilePredictionSplit <- QuantilePrediction[,.(Split = unique(Split)),Interval][order(as.character(Interval))]
QuantilePredictionArray <- GenQuantilePredictionArray(QuantilePrediction)
TrainRows <- QuantilePredictionSplit[,which(Split=="Train")]
TestRows <- QuantilePredictionSplit[,which(Split=="Test")]
ValidationRows <- QuantilePredictionSplit[,which(Split=="Validation")]

ReturnArray <- ReturnArray[dimnames(QuantilePredictionArray)[[1]],,]
ReturnArray[is.na(ReturnArray)]=0

# dim(QuantilePredictionArray)[[1]] - length(TrainRows) - length(TestRows) - length(ValidationRows)
# all(QuantilePredictionSplit$Interval == dimnames(ReturnArray)[[1]])
# all(QuantilePredictionSplit$Interval == dimnames(QuantilePredictionArray)[[1]])

y <- torch_tensor(ReturnArray)
x <- torch_tensor(QuantilePredictionArray)
x <- x$nan_to_num(-1)
y_train <- y[TrainRows,]
x_train <- x[TrainRows,]
y_test <- y[TestRows,]
x_test <- x[TestRows,]
y_validation <- y[ValidationRows,]
x_validation <- x[ValidationRows,]
criterion = function(y_pred, y) {
  RPS <- ComputeSharpTensor(weights = y_pred, y)
  -RPS$mean()
}



# baseModel ---------------------------------------------------------------


constructModel = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
  },
  forward = function(x) {
    invalidStocks <- torch_einsum("nkf->nk",list(x[,,1:5] == -1)) > 0
    for(i in seq_along(self$layerSizes)){
      x <- self[[str_c("layer_",i)]](x)
      # x[1,69:71,]
      if (i == length(self$layerSizes)){
        x <- torch_squeeze(x,3)
        x[invalidStocks] = -Inf
      }
      x <- self$layerTransforms[[i]](x)
    }
    x
  }
)


inputSize <- dim(x)[3]
layerSizes <- c(1)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {
  nnf_softmax(x,2)
  # x
  # nnf_softmax(torch_squeeze(x,3),2)
  # nnf_softmax(torch_squeeze(x,3)$nan_to_num(-Inf),2)
  # nnf_softmax(torch_squeeze(x,3)$nan_to_num(0),2)
}))
# model <- constructFFNN(inputSize, layerSizes, layerTransforms)
model <- constructModel(inputSize, layerSizes, layerTransforms)

train <- list(y_train, x_train)
test <- list(y_test, x_test)



fit <- trainModel(model = model, train, test, criterion, epochs = 100, minibatch = Inf, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
model <- fit$model
progress <- fit$progress











x_debug <- x_test[67:106,,]
x_debug <- x_debug$nan_to_num(-1)
y_debug <- y_test[67:106,,]
x <- x_debug
self <- list()
optimizer = optim_adam(model$parameters, lr = 0.01)

model$state_dict()

optimizer$zero_grad()
y_pred <- model(x_debug)
loss = criterion(y_pred, y_debug)
loss
loss$backward()
optimizer$step()


model$state_dict()




















































if(T){
  start <- Sys.time()
 
  Sys.time() - start 
  baseModel <- fit$model
  baseModelProgress <- fit$progress
  saveRDS(baseModelProgress, file.path("Precomputed","baseModelProgress.RDS"))
  torch_save(baseModel, file.path("Precomputed", str_c("baseModel", ".t7")))
}else{
  baseModelProgress <- readRDS(file.path("Precomputed","baseModelProgress.RDS"))
  baseModel <- torch_load(file.path("Precomputed", str_c("baseModel", ".t7")))
}



inputSize <- dim(x)[3]
layerSizes <- c(10, 1)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {
  nnf_softmax(torch_squeeze(x,3)$nan_to_num(-Inf),2)
}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms)

weights <- baseModel(x_train)
criterion(weights, y_train)











out <- computeSharpTensor(weights, y_train)$mean()
out$mean()





weights <- baseModel(x[418:419,,])
# dim(weights)
# sum(weights[2,])
# 
ysub <- y[418:419,,]



# rows <- 1668:1669
# rows <- 1500:1501
rows <- 9:10
weights <- baseModel(x[rows,,])
y <- y[rows,,]
computeSharpTensor(y,weights)

weights <- baseModel(x)
out <- computeSharpTensor(y,weights)
which(is.na(as.array(out)))

# weights$sum(2)
as.array(weights)
as.array(x[rows,,])


temp <- matrix(c(1:5,rep(1,5)),nrow=5) + 0.01
temp[5,] <- NA
temp <- torch_tensor(temp)
temp <- temp$nan_to_num(-Inf)
nnf_softmax(temp,1)


ret - sret/20

sret[]
torch_sub(ret,sret[,])


sdp <- torch_std(ret,dim=2, unbiased=TRUE)



sum(ysub[2,,7] * weights[2,])



computeSharpTensor <- function(x, weights) {
  RET <- torch_squeeze(torch_einsum("mnp,nk->mkp",list(x-1,weights)),dim=2)
  ret <- torch_log(1 + RET)
  sret <- torch_einsum("mn->n",ret)
  Tobs <- ret$size()[1]
  sdp <- torch_std(ret,dim=1, unbiased=TRUE)
  ((21*12) / sqrt(252)) * (1/Tobs) * sret/sdp
}








temp <- as.array(nnf_softmax(baseModel(x[1668:1669,,]),2))




y[1668:1669,]


all(dimnames(QuantilePredictionArray)[[1]] %in% dimnames(ReturnArray)[[1]])


x <- torch_tensor(QuantilePredictionArray)






























# plot(Stocks[["VXX"]]$VXX.Adjusted)

QuantilePrediction <- QuantilePredictions$meta

# temp <- QuantilePrediction[Split == "Test"][order(Rank5 - Rank1)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# View(temp)
# temp <- QuantilePrediction[Split == "Test"][order(Rank5 + Rank4 - Rank1 - Rank2)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# View(temp)
submission <- QuantilePrediction[Split=="Validation"]
submission <- submission[match(template$ID, submission$Ticker), .(ID = Ticker, Rank1 , Rank2, Rank3, Rank4, Rank5, Decision = 0)]
# View(submission[order(Rank5 - Rank1)])
# View(submission[order(Rank5 + Rank4 - Rank1 - Rank2)])
submission[ID == "VXX", Decision := -.125]
submission[ID == "V", Decision := .0625]
submission[ID == "CDW", Decision := .0625]


submission <- validateSubmission(submission, Round = T)

write.csv(submission,file.path("Results","SubmissionManual.csv"),row.names = F, quote=FALSE)




