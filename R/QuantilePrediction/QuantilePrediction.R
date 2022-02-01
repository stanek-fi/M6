library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
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

TrainStart <- as.Date("1980-01-07")
TrainEnd <- as.Date("2020-01-12")
ValidationStart <- as.Date("2021-12-13")
ValidationEnd <- as.Date("2022-01-09")
# ValidationStart <- as.Date("2022-01-10")
# ValidationEnd <- as.Date("2022-02-06")
# ValidationStart <- as.Date("2022-02-07")
# ValidationEnd <- as.Date("2022-03-06")

s <- 1
Stocks <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
  Stock <- Stocks[[s]]
  Ticker <- names(Stocks)[s]
  colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
  Stock <- AugmentStock(Stock[index>=TrainStart], ValidationEnd)
  Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
  Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
  Stock[,Ticker := Ticker]
  Stock
}))

featureList <- c(
  list(
    function(SD) {Return(SD)}, #first is just return for y generation
    function(SD) {LagVolatility(SD, lags = 1:5)},
    function(SD) {LagReturn(SD, lags = 1:5)}
  ),
  TTR
)
Sanitize <- T
if(Sanitize){
  StocksAggrIn <- Stocks[index < ValidationStart,computeFeatures(.SD,featureList),.(Ticker)]
  StocksAggrAll <- Stocks[,computeFeatures(.SD,featureList),.(Ticker)]
  StocksAggr <- rbind(StocksAggrIn,StocksAggrAll[as.Date(str_sub(Interval,1,20))>=ValidationStart])
}else{
  StocksAggr <- Stocks[,computeFeatures(.SD,featureList),.(Ticker)]
}
featureNames <- names(StocksAggr)[-(1:3)]
StocksAggr[, ReturnQuintile := computeQuintile(Return), Interval]
StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr[, IntervalStart := as.Date(str_sub(Interval,1,10))]
StocksAggr[, IntervalEnd := as.Date(str_sub(Interval,14,23))]
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]

TrainRows <- which(StocksAggr[,(IntervalStart >= TrainStart) & (IntervalEnd <= TrainEnd)])
TestRows <- which(StocksAggr[,(IntervalStart > TrainEnd) & (IntervalEnd < ValidationStart)])
ValidationRows <- which(StocksAggr[,(IntervalStart >= ValidationStart) & (IntervalEnd <= ValidationEnd)])

y <- StocksAggr[,ReturnQuintile]
y <- torch_tensor(t(sapply(y,function(x) {
  if(is.na(x)){
    rep(NA,5)
  }else{
    replace(numeric(5), x:5, 1)
  }
})), dtype = torch_float())

x <- StocksAggr[,.SD,.SDcols = featureNames]
x <- torch_tensor(as.matrix(x), dtype = torch_float())

xtype_factor <- as.factor(StocksAggr$Ticker)
i <- torch_tensor(t(cbind(seq_along(xtype_factor),as.integer(xtype_factor))),dtype=torch_int64())
v <- torch_tensor(rep(1,length(xtype_factor)))
xtype <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor),length(levels(xtype_factor))))$coalesce()


y_train <- y[TrainRows,]
x_train <- x[TrainRows,]
xtype_train <- subsetTensor(xtype, rows = TrainRows)
y_test <- y[TestRows,]
x_test <- x[TestRows,]
xtype_test <- subsetTensor(xtype, rows = TestRows)
y_validation <- y[ValidationRows,]
x_validation <- x[ValidationRows,]
xtype_validation <- subsetTensor(xtype, rows = ValidationRows)
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

y_pred <- baseModel(x_validation)
ComputeRPSTensor(y_pred,y_validation)

# metaModel ---------------------------------------------------------------

metaModel <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1)
minibatch <- function() {minibatchSampler(10,xtype_train)}
train <- list(y_train, x_train, xtype_train)
test <- list(y_test, x_test, xtype_test)

start <- Sys.time()
fit <- trainModel(model = metaModel, train, test, criterion, epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
Sys.time() - start 
metaModel <- fit$model
metaModelProgress <- fit$progress


temp <- rbind(
  melt(baseModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,type := "base"],
  melt(metaModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,epoch := epoch + baseModelProgress[,which.min(loss_test)]][,type := "meta"]
)
ggplot(temp, aes(x = epoch, y = value, colour =variable, linetype = type))+
  geom_line()

y_pred <- metaModel(x_validation, xtype_validation)
ComputeRPSTensor(y_pred,y_validation)

# cbind(as.array(y_validation),as.array(y_pred))

# mesaModel ---------------------------------------------------------------

mesaModel <- metaModel$MesaModel(metaModel)()
j <- 82
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



