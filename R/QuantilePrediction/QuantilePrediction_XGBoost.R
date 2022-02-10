library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
library(xgboost)
rm(list=ls())
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

# Shifts <- c(0)
Shifts <- c(0,7,14,21)
# Shifts <- c(0,7)
Submission = 0
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)


GenerateStockAggr <- F
if(GenerateStockAggr){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
  # Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  # SP500Tickers <- sample(StockNames[SP500==TRUE,Symbol],100)
  # ETFTickers <- sample(StockNames[ETF==TRUE,Symbol],0)
  # Stocks <- Stocks[c(SP500Tickers,ETFTickers)]
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, CheckLeakage = F)
  saveRDS(StocksAggr, file.path("Precomputed","StocksAggr.RDS"))
}else{
  StocksAggr <- readRDS(file.path("Precomputed","StocksAggr.RDS"))
}



featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return", "Shift", "ReturnQuintile", "IntervalStart", "IntervalEnd"))]
StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]


# TrainStart <- as.Date("1990-01-01")
TrainStart <- as.Date("1990-01-01")
TrainEnd <- as.Date("2019-01-01")
# TrainEnd <- as.Date("2021-01-01")
ValidationStart <- as.Date("2021-01-01")
# ValidationStart <- as.Date("2021-12-13")
# ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - 1]
ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - 1]

TrainRows <- which(StocksAggr[,(IntervalStart >= TrainStart) & (IntervalEnd <= TrainEnd)])
TestRows <- which(StocksAggr[,(IntervalStart > TrainEnd) & (IntervalEnd < ValidationStart)])
ValidationRows <- which(StocksAggr[,(IntervalStart >= ValidationStart) & (IntervalEnd <= ValidationEnd)])





y <- StocksAggr[,ReturnQuintile]
yT <- torch_tensor(t(sapply(y,function(x) {
  if(is.na(x)){
    rep(NA,5)
  }else{
    replace(numeric(5), x:5, 1)
  }
})), dtype = torch_float())
x <- StocksAggr[,.SD,.SDcols = featureNames]

y_train <- y[c(TrainRows,TestRows)]
x_train <- x[c(TrainRows,TestRows),]
train_matrix <- xgb.DMatrix(data = as.matrix(x_train), label = y_train-1)

y_validation <- y[ValidationRows]
x_validation <- x[ValidationRows,]
validation_matrix <- xgb.DMatrix(data = as.matrix(x_validation), label = y_validation-1)
yT_validation <- yT[ValidationRows,]

xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = 5)

# start <- Sys.time()
# cv <- xgb.cv(params = xgb_params, data = train_matrix, nrounds = 5, nfold=5)
# Sys.time() - start

start <- Sys.time()
fit <- xgb.train(params = xgb_params, data = train_matrix, nrounds = 50)
Sys.time() - start

test_pred <- predict(fit, newdata = validation_matrix)
test_pred <- matrix(test_pred,ncol=5, byrow = T)
y_pred_XGB <- torch_tensor(test_pred )

as.array(ComputeRPSTensor(y_pred_XGB,yT_validation))

importance <- xgb.importance(model = fit)
importance









