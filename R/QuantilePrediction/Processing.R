library(data.table)
library(stringr)
library(torch)
rm(list=ls())

Stocks <- readRDS(file.path("Data","StocksSP500.RDS"))

TimeEnd <- as.Date("2023-01-08")
TimeStart <- TimeEnd - (7*4) * 1000
TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
TimeBreaksNames <- str_c(TimeBreaks[-length(TimeBreaks)]+1, " : " , TimeBreaks[-1])
  
StocksAggr <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
  Stock <- Stocks[[s]]
  StockName <- names(Stocks)[s]
  colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")             
  Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
  Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
  Stock[,.(Return = last(Adjusted)/first(Adjusted) - 1, StockName = StockName), Interval]
}))

computeQuintile <- function(x){
  findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
} 

StocksAggr[,table(Interval)]
StocksAggr[,ReturnQuintile := computeQuintile(Return), Interval]



















View(StocksAggr[Interval=="2021-11-15 : 2021-12-12"][order(Return)])
StocksAggr[,sum(ReturnQuintile==4),Interval][order(Interval)]



x <- StocksAggr[Interval=="2021-11-15 : 2021-12-12",Return]

rank(x)

rank(x)
findInterval(rank(x),)
table(findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T))

temp[Interval=="2021-10-18 : 2021-11-14"]
temp[Interval=="2021-11-15 : 2021-12-12"]
View(temp[.(index,int)])


findInterval(temp$index,TimeBreaks,left.open=T)


plot(temp$MMM.Close)
     


x=1:10
cbind(x,findInterval(x, c(0,3,8,10),left.open=T))