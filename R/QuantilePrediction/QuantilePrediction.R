library(data.table)
library(stringr)
library(torch)
rm(list=ls())
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")

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
  function(SD) {LagVolatility(SD, lags = 1:2)},
  function(SD) {LagReturn(SD, lags = 1:2)}
)
StocksAggr <- Stocks[,computeFeatures(.SD,featureList),.(Ticker)]
featureNames <- names(StocksAggr)[-(1:3)]
StocksAggr[, ReturnQuintile := computeQuintile(Return), Interval]

StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)


y <- StocksAggr[,ReturnQuintile]
x <- StocksAggr[,.SD,.SDcols = featureNames]


model <- FFNN(x,y)


x_train = torch_tensor(as.matrix(x), dtype = torch_float())
y_train = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())
y_pred = model(x_train)
ComputeRPSTensor(y_pred,y_train)
ComputeRPSTensor(torch_tensor(matrix(0.2,ncol=5,nrow=length(y)), dtype = torch_float()),y_train)





library()

