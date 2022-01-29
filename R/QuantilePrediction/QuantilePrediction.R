library(data.table)
library(stringr)
library(torch)
library(ggplot2)
rm(list=ls())
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")

Stocks <- readRDS(file.path("Data","StocksM6.RDS"))

TimeEnd <- as.Date("2023-01-08")
TimeStart <- TimeEnd - (7*4) * 1000
TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
TimeBreaks <- TimeBreaks[TimeBreaks>as.Date("1969-12-01")]
TimeBreaksNames <- str_c(TimeBreaks[-length(TimeBreaks)]+1, " : " , TimeBreaks[-1])
  
s <- 1
Stocks <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
  Stock <- Stocks[[s]]
  Ticker <- names(Stocks)[s]
  colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")             
  Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
  Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
  Stock[,Ticker := Ticker]
  Stock
}))

featureList <- list(
  function(SD) {Return(SD)}, #first is just return for y generation
  function(SD) {LagVolatility(SD, lags = 1:5)},
  function(SD) {LagReturn(SD, lags = 1:5)}
)
StocksAggr <- Stocks[,computeFeatures(.SD,featureList),.(Ticker)]
featureNames <- names(StocksAggr)[-(1:3)]
StocksAggr[, ReturnQuintile := computeQuintile(Return), Interval]

StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr[, IntervalStart := as.Date(str_sub(Interval,1,10))]
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]
StocksAggr <- StocksAggr[Ticker %in% unique(StocksAggr$Ticker)[1:100]]

y <- StocksAggr[,ReturnQuintile]
y = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())

x <- StocksAggr[,.SD,.SDcols = featureNames]
x = torch_tensor(as.matrix(x), dtype = torch_float())

xtype_factor <- as.factor(StocksAggr$Ticker)
i <- torch_tensor(t(cbind(seq_along(xtype_factor),as.integer(xtype_factor))),dtype=torch_int64())
v <- torch_tensor(rep(1,length(xtype_factor)))
xtype <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor),length(levels(xtype_factor))))$coalesce()


trainSplit <- 0.95
trainN <- round(nrow(x)*trainSplit)
trainRows <- 1:trainN
testRows <- (trainN+1):nrow(x)

y_train <- y[trainRows,]
x_train <- x[trainRows,]
xtype_train <- subsetTensor(xtype, rows = trainRows)
y_test <- y[testRows,]
x_test <- x[testRows,]
xtype_test <- subsetTensor(xtype, rows = testRows)
criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}


# baseModel ---------------------------------------------------------------

inputSize <- length(featureNames)
layerSizes <- c(32, 8, 5)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {nnf_softmax(x,2)}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms)
baseModel = prepareBaseModel(baseModel,x = x_train)
train <- list(y_train, x_train)
test <- list(y_test, x_test)

start <- Sys.time()
fit <- trainModel(model = baseModel, train, test, criterion, epochs = 100, minibatch = Inf, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
Sys.time() - start 
baseModel <- fit$model
baseModelProgress <- fit$progress


# metaModel ---------------------------------------------------------------

metaModel <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1)
minibatch <- function() {minibatchSampler(100,xtype_train)}
train <- list(y_train, x_train, xtype_train)
test <- list(y_test, x_test, xtype_test)

start <- Sys.time()
fit <- trainModel(model = metaModel, train, test, criterion, epochs = 20, minibatch = minibatch, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
Sys.time() - start 
metaModel <- fit$model
metaModelProgress <- fit$progress


temp <- rbind(
  melt(baseModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,type := "base"],
  melt(metaModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,epoch := epoch + baseModelProgress[,which.min(loss_test)]][,type := "meta"]
)
ggplot(temp, aes(x = epoch, y = value, colour =variable, linetype = type))+
  geom_line()



# mesaModel ---------------------------------------------------------------

mesaModel <- metaModel$MesaModel(metaModel)()
j <- 83
rows_train <- xtype_train$indices()[2,] == (j-1)
x_train_subset <- x_train[rows_train,]
y_train_subset <- y_train[rows_train]
rows_test <- xtype_test$indices()[2,] == (j-1)
x_test_subset <- x_test[rows_test,]
y_test_subset <- y_test[rows_test]
train <- list(y_train_subset, x_train_subset)
test <- list(y_test_subset, x_test_subset)

start <- Sys.time()
fit <- trainModel(model = mesaModel, train, test, criterion, epochs = 50, minibatch = Inf, tempFilePath = NULL, patience = Inf, printEvery = 10)
Sys.time() - start 
mesaModel <- fit$model
mesaModelProgress <- fit$progress

as.array(mesaModel$state_dict())
t(as.array(metaModel$state_dict()$mesaLayerWeight))[max(1,(j-1)):(j+1)]



