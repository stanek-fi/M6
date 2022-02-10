library(quantmod)
library(data.table)
require(TTR)
library(BatchGetSymbols)
library(stringr)
library(imputeTS)
set.seed(1)
source("R/GenData/GenData_Helpers.R")

StockNames <- readRDS(file.path("Data", "StockNames.RDS"))
Stocks <- readRDS(file.path("Data","StocksM6.RDS"))



s <- 1

for(s in seq_along(Stocks)) {
  colnamesStock <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
  if(s %% 1 == 0){
    print(str_c("Stock:", s, " Time:", Sys.time()))
  }
  Stock <- Stocks[[s]]
  Ticker <- names(Stocks)[s]
  colnames(Stock) <- colnamesStock
  
}



i <- 1
Stocks <- lapply(seq_along(tickers), function(i) {
  print(round(i/length(tickers),3))
  ticker <- tickers[i]
  out <- try({
    as.data.table(getSymbols(ticker,from = from,to = to,auto.assign=FALSE))
  })
  if("try-error" %in% class(out)){
    out <- NULL
  }
  return(out)
})




aux <- rep(NA,10)
test <- lapply(1:10, function(i){
  aux[i] <<- 3
  i+1
})
