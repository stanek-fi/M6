library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls());gc()
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")


featureList <- c(
  list(
    function(SD, BY) {Return(SD)}, #first is just return for y generation
    function(SD, BY) {LagVolatility(SD, lags = 1:7)},
    function(SD, BY) {LagReturn(SD, lags = 1:7)},
    function(SD, BY) {IsETF(SD, BY, StockNames = StockNames)}
  )
  # TTR
)

# Shifts <- c(0)
Shifts <- c(0,7,14,21)
# Shifts <- c(0,7)
Submission = 10
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)

GenerateStockAggr <- TRUE
if(GenerateStockAggr){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  StockNames[Symbol=="FB",Symbol := "META"]  # * needed to account for the fact that one cannot download FB data under FB ticker anymore
  StockNames <- StockNames[Symbol!="ANAT"]  # * not quite sure why I need to drop this
  Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  # temp <- StockNames[M6Dataset>0 & M6Dataset<=2][order(M6Dataset),.(Symbol,M6Dataset)]
  # temp <- StockNames[M6Dataset>0][order(M6Dataset),.(Symbol,M6Dataset)]
  # temp <- StockNames[M6Dataset>0 & Symbol != "JW-A"][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  # temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  # temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS", "ANAT", "ENIA"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  # temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS", "ANAT", "ENIA", "ACC"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS", "ANAT", "ENIA", "ACC", "LFC", "SHI"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  Stocks <- Stocks[temp$Symbol]
  ZeroedStocks <- temp[,.SD[1],M6Dataset][,Symbol]
  ZeroedStock  <- ZeroedStocks[1]
  for (ZeroedStock in ZeroedStocks) {
    Stocks[[ZeroedStock]][,Adjusted := 1]
  }
  M6Datasets <- temp$M6Dataset
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, M6Datasets, CheckLeakage = F)
  # StocksAggr <- GenStocksAggr(Stocks[800:1000], IntervalInfos, featureList, M6Datasets[800:1000], CheckLeakage = F)
  saveRDS(StocksAggr, file.path("Precomputed","StocksAggr_ZeroReturnCorrection.RDS"))
}else{
  StocksAggr <- readRDS(file.path("Precomputed","StocksAggr_ZeroReturnCorrection.RDS"))
}

# StocksAggr[Shif@t == 0 & M6Dataset == 1]

featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return", "Shift", "M6Dataset", "ReturnQuintile", "IntervalStart", "IntervalEnd"))]
StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
# StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames[!(featureNames %in% c("ETF"))])
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]



TrainStart <- as.Date("2000-01-01")
# TrainStart <- as.Date("2010-01-01")
# TrainEnd <- as.Date("2020-01-01")
TrainEnd <- as.Date("2022-01-01")
# TrainEnd <- as.Date("2022-03-01")
# ValidationStart <- as.Date("2021-01-01")
# ValidationEnd <- as.Date("2022-01-01")
q <- 0
ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - q]
ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - q]



freq <- table(StocksAggr[Ticker %in% ZeroedStocks, ReturnQuintile])
ZeroReturnCorrection <- freq/sum(freq)

saveRDS(ZeroReturnCorrection, file.path("Precomputed","ZeroReturnCorrection.RDS"))

