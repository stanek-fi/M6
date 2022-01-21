library(quantmod)
library(data.table)
require(TTR)
library(BatchGetSymbols)
library(stringr)


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
print("recent dates:")
table(as.Date(sapply(Stocks, function(s) s[.N,index])))



invalidIndices <- "IEFM.L"
ii <- invalidIndices[1]
plot(Stocks[[ii]]$IEFM.L.Adjusted)
for(ii in invalidIndices){ #works only for consecutive error period
  temp <- copy(Stocks[[ii]])
  invalidRange <- temp[get(str_c(ii,".Adjusted"))<100, index]
  startPrice <- temp[invalidRange[1]>index,][,last(na.omit(get(str_c(ii,".Adjusted"))))]
  endPrice <- temp[invalidRange[length(invalidRange)]<index,][,first(na.omit(get(str_c(ii,".Adjusted"))))]
  subPrice <- temp[invalidRange[1]>index,][(.N - length(invalidRange) + 1):(.N),get(str_c(ii,".Adjusted"))]
  subPrice <- subPrice + (startPrice - subPrice[1]) 
  subPrice <- subPrice + (seq_along(invalidRange)-1)/(length(invalidRange)) *  (endPrice - subPrice[length(subPrice)]) 
  temp[index %in% invalidRange, IEFM.L.Adjusted := subPrice]
  Stocks[[ii]] <- temp
}
plot(Stocks[[ii]]$IEFM.L.Adjusted)


saveRDS(Stocks,file.path("Data","StocksM6.RDS"))

