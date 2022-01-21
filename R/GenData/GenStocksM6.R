library(quantmod)
library(data.table)
require(TTR)
library(BatchGetSymbols)
library(stringr)
library(imputeTS)
set.seed(1)
source("R/GenData/GenData_Helpers.R")

StockNames <- readRDS(file.path("Data", "StockNames.RDS"))
tickers <- StockNames[!is.na(M6Id), Symbol]

from = "1970-01-01"
to = "2023-12-01"
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
names(Stocks)=tickers
Stocks <- Stocks[sapply(Stocks, function(y) {!is.null(y)})]


if(!all(names(Stocks) == tickers)){
  warning("Some stock missing")
}
table(as.Date(sapply(Stocks, function(s) s[.N,index])))


plot(Stocks[["IEFM.L"]]$IEFM.L.Adjusted) #manually correcting invalid data
Stocks[["IEFM.L"]][IEFM.L.Adjusted<100,IEFM.L.Adjusted := NA]


StocksClean<- setNames(lapply(names(Stocks), function(ticker) {
  stock <- Stocks[[ticker]]
  naRows <- sum(apply(is.na(stock),1,any))
  if(naRows>0){
    print(str_c("Ticker: ",ticker, " Missing: ", naRows))
    return(cbind(stock[,.(index)],stock[,lapply(.SD, noisyInterpolation), .SDcols=names(stock)[-1]]))
  }else{
    return(stock)
  }
}),names(Stocks))

saveRDS(StocksClean,file.path("Data","StocksM6.RDS"))