rm(list=ls());gc()
library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")

featureList <- c(
  list(
    function(SD, BY) {Return(SD)}, #first is just return for y generation
    function(SD, BY) {LagVolatility(SD, lags = 1:5)},
    function(SD, BY) {LagReturn(SD, lags = 1:5)},
    function(SD, BY) {IsETF(SD, BY, StockNames = StockNames)}
  ),
  TTR
)

Shifts <- c(0,7,14,21)
Submission = 0
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)

GenerateStockAggr <- T
if(GenerateStockAggr){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  # temp <- StockNames[M6Dataset>0 & M6Dataset<=2][order(M6Dataset),.(Symbol,M6Dataset)]
  temp <- StockNames[M6Dataset>0][order(M6Dataset),.(Symbol,M6Dataset)]
  Stocks <- Stocks[temp$Symbol]
  M6Datasets <- temp$M6Dataset
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, M6Datasets, CheckLeakage = F)
  saveRDS(StocksAggr, file.path("Precomputed","StocksAggr.RDS"))
}else{
  StocksAggr <- readRDS(file.path("Precomputed","StocksAggr.RDS"))
}
featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return", "Shift", "M6Dataset", "ReturnQuintile", "IntervalStart", "IntervalEnd"))]

# featureNames <- c(
#   str_c("VolatilityLag", 1:5),  str_c("ReturnLag", 1:5),
#   c("ETF", "volatility_n=100_last", "volatility_n=100", "volatility_(garman.klass)", "volatility_n=10", "ATR_tr_n=28", "ADX1_DIp", "ADX1_DIn", "ATR_atr_n=28", "ATR_trueHigh_n=28", "RSI",  "SMI2_signal_n=13", "SMI2_signal_n=30", "TRIX1_TRIX_n=40", "TRIX1_signal_n=40", "BBands2_dn", "BBands2_up", "BBands2_pctB")
# )

# importance_train <- readRDS(file.path("Precomputed", "importance_train.RDS"))
# importance_test <- readRDS(file.path("Precomputed", "importance_test.RDS"))
# temp <- data.table(featureNames,importance_train = rowMeans(do.call(cbind,importance_train)), importance_test = rowMeans(do.call(cbind,importance_test)))
# temp <- temp[order(importance_test,decreasing = T)][,position_test := 1:.N]
# temp <- temp[order(importance_train,decreasing = T)][,position_train := 1:.N]
# # temp[str_detect(featureNames, "ADX")]
# featureNames <- temp[order(importance_test,decreasing = T)][1:50,featureNames]

StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- StocksAggr[order(Ticker,IntervalStart)]
# StocksAggr <- StocksAggr[M6Dataset==1]
# StocksAggr <- StocksAggr[ETF>0]
# StocksAggrOld <-StocksAggr
# featureNamesOld <- featureNames 
# 
# 
# StocksAggrOld[,.N,Ticker][order(N)]
# # StocksAggr <- StocksAggrOld[Ticker == "XOM"]
# StocksAggr <- StocksAggrOld[Ticker == "RTX"]
# StocksAggr <- StocksAggrOld[Ticker == "XOM"]
# StocksAggr <- StocksAggrOld[Ticker %in% c("XOM","RTX")]
# featureNames <- featureNamesOld[6:10]

TrainStart <- as.Date("2000-01-01")
TrainEnd <- as.Date("2020-01-01")
ValidationStart <- as.Date("2021-01-01")
ValidationEnd <- as.Date("2022-01-01")
# ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - 1]
# ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - 1]

TrainRows <- which(StocksAggr[,(IntervalStart >= TrainStart) & (IntervalEnd <= TrainEnd)])
TrainInfo <- StocksAggr[TrainRows,.(Interval, IntervalStart, IntervalEnd, Shift, M6Dataset, Ticker, Return)]
TestRows <- which(StocksAggr[,(IntervalStart > TrainEnd) & (IntervalEnd < ValidationStart)])
TestInfo <- StocksAggr[TestRows,.(Interval, IntervalStart, IntervalEnd, Shift, M6Dataset, Ticker, Return)]
ValidationRows <- which(StocksAggr[,(IntervalStart >= ValidationStart) & (IntervalEnd <= ValidationEnd)])
ValidationInfo <- StocksAggr[ValidationRows,.(Interval, IntervalStart, IntervalEnd, Shift, M6Dataset, Ticker, Return)]

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
xtype_factor_M6Dataset <- StocksAggr[,.(M6Dataset = unique(M6Dataset)),Ticker][match(levels(xtype_factor),Ticker)]
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


rm(StocksAggr)
gc()
# testing perf -----------------------------------------------------------------
# train <- list(y_train, x_train)
# test <- list(y_test, x_test)
# validation <- list(y_validation, x_validation)

R <- 3
res <- rep(NA,R)
resM6 <- matrix(NA,R,10)
importance_train <- vector(mode="list",R)
importance_test <- vector(mode="list",R)
importance_validation <- vector(mode="list",R)
start <- Sys.time()
r <- 1

for(r in 1:R){
  set.seed(r)
  torch_manual_seed(r)
  print(r)
  inputSize <- length(featureNames)
  layerSizes <- c(32, 8, 5)
  # layerSizes <- c(16,8, 5)
  # layerSizes <- c(6,5)
  layerDropouts <- c(rep(0.2, length(layerSizes)-1),0)
  layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_leaky_relu), list(function(x) {nnf_softmax(x,2)}))
  baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
  # baseModel = prepareBaseModel(baseModel,x = x_train)
  minibatch = c(128, rep(1000, 5), rep(5000,100))
  # minibatch = c(64,rep(1000,100))
  # minibatch = 64
  # minibatch = c(64, rep(1000, 5), rep(2000, 5), rep(5000, 10), rep(10000, 100))
  # minibatch <- function() {minibatchSampler(20,xtype_train)}
  lr <- 0.001
  if(T){
    start <- Sys.time()
    fit <- trainModel(model = baseModel, criterion, train = list(y_train, x_train), test = list(y_test, x_test), validation = list(y_validation, x_validation), epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr=lr)
    # fit <- trainModel(model = baseModel, criterion, train = list(y_train, x_train), test = list(y_test, x_test), validation = NULL, epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = Inf, lr=lr)
    Sys.time() - start
    baseModel <- fit$model
    baseModelProgress <- fit$progress
    saveRDS(baseModelProgress, file.path("Precomputed","baseModelProgress.RDS"))
    torch_save(baseModel, file.path("Precomputed", str_c("baseModel", ".t7")))
  }else{
    baseModelProgress <- readRDS(file.path("Precomputed","baseModelProgress.RDS"))
    baseModel <- torch_load(file.path("Precomputed", str_c("baseModel", ".t7")))
  }
  y_pred_base <- baseModel(x_validation)
  loss_validation_base <- as.array(ComputeRPSTensor(y_pred_base,y_validation))
  loss_validation_base_vector <- as.array(ComputeRPSTensorVector(y_pred_base,y_validation))
  loss_validation_base_M6Dataset <- sapply(1:max(ValidationInfo$M6Dataset), function(i) {mean(loss_validation_base_vector[which(ValidationInfo$M6Dataset == i)])})

  # importance_train[[r]] <- evaluateImportance(x_train,function(x) {as.array(ComputeRPSTensor(baseModel(x),y_train))})
  # importance_test[[r]] <- evaluateImportance(x_test,function(x) {as.array(ComputeRPSTensor(baseModel(x),y_test))})
  # importance_validation[[r]] <- evaluateImportance(x_validation,function(x) {as.array(ComputeRPSTensor(baseModel(x),y_validation))})
  res[r] <- loss_validation_base
  resM6[r,] <- loss_validation_base_M6Dataset
}
Sys.time()-start

resM6
res
mean(res)

# saveRDS(importance_train, file.path("Precomputed", "importance_train.RDS"))
# saveRDS(importance_test, file.path("Precomputed", "importance_test.RDS"))
# saveRDS(importance_validation, file.path("Precomputed", "importance_validation.RDS"))

# .rs.restartR()

