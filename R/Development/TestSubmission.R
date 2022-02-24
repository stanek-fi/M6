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
source("R/PortfolioOptimizationNN/PortfolioOptimizationNN_Helpers.R")

featureList <- c(
  list(
    function(SD, BY) {Return(SD)}, #first is just return for y generation
    function(SD, BY) {LagVolatility(SD, lags = 1:5)},
    function(SD, BY) {LagReturn(SD, lags = 1:5)},
    function(SD, BY) {IsETF(SD, BY, StockNames = StockNames)}
  )
)

Shifts <- c(0)
Submission = 0
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)

GenerateStockAggr <- T
if(GenerateStockAggr){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  # temp <- StockNames[M6Dataset>0 & M6Dataset<=2][order(M6Dataset),.(Symbol,M6Dataset)]
  temp <- StockNames[M6Dataset==1][order(M6Dataset),.(Symbol,M6Dataset)]
  Stocks <- Stocks[temp$Symbol]
  M6Datasets <- temp$M6Dataset
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, M6Datasets, CheckLeakage = F)
  # saveRDS(StocksAggr, file.path("Precomputed","StocksAggr.RDS"))
}else{
  # StocksAggr <- readRDS(file.path("Precomputed","StocksAggr.RDS"))
}
featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return", "Shift", "M6Dataset", "ReturnQuintile", "IntervalStart", "IntervalEnd"))]



submission <- as.data.table(read.csv(file.path("Results","SubmissionManual.csv")))


StocksAggrSelected <- StocksAggr[Interval == "2022-02-07 : 2022-03-06", .(Ticker, Interval, Return, ReturnQuintile)]
StocksAggrSelected <- StocksAggrSelected[match(submission$ID,Ticker)]

cbind(submission, StocksAggrSelected)

y <- StocksAggrSelected[,ReturnQuintile]
y <- torch_tensor(t(sapply(y,function(x) {
  if(is.na(x)){
    rep(NA,5)
  }else{
    replace(numeric(5), x:5, 1)
  }
})), dtype = torch_float())
y_pred <- torch_tensor(as.matrix(submission[,.(Rank1, Rank2, Rank3, Rank4, Rank5)]))
ComputeRPSTensor(y_pred,y)
















y_pred <- torch_tensor(as.matrix(submission[,.(Decision)]))
ComputeSharpTensor(weights = y_pred, train[[1]])






























