library(quantmod)
library(data.table)
require(TTR)
library(BatchGetSymbols)

temp <- stockSymbols()
tickers <- temp[[1]]
# tickers = c("AAPL", "sdfkjsldfk", "NFLX", "AMZN", "K", "O")


from = "1900-01-01"
to = "2021-12-01"
Stocks <- lapply(seq_along(tickers), function(i) {
  print(round(i/length(tickers),3))
  ticker <- tickers[i]
  out <- try({
    as.data.table(getSymbols(ticker,from = from,to = to,auto.assign=FALSE))
  })
  if(class(out) == "try-error"){
    out <- NULL
  }
  return(out)
})
names(Stocks)=tickers
Stocks <- Stocks[sapply(Stocks, function(y) {!is.null(y)})]
saveRDS(Stocks,file.path("Data","Stocks.RDS"))
# Stocks <- readRDS(file.path("Data","Stocks.RDS"))


df.SP500 <- GetSP500Stocks()
tickers <- df.SP500$Tickers

from = "1900-01-01"
to = "2021-12-01"
Stocks <- lapply(seq_along(tickers), function(i) {
  print(round(i/length(tickers),3))
  ticker <- tickers[i]
  out <- try({
    as.data.table(getSymbols(ticker,from = from,to = to,auto.assign=FALSE))
  })
  if(class(out) == "try-error"){
    out <- NULL
  }
  return(out)
})
names(Stocks)=tickers
Stocks <- Stocks[sapply(Stocks, function(y) {!is.null(y)})]
saveRDS(Stocks,file.path("Data","StocksSP500.RDS"))



