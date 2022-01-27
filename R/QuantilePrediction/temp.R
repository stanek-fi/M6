library(data.table)
library(stringr)
library(torch)
library(ggplot2)
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


library(TTR)
library(fTrading)


# SD <- Stocks[Ticker == "AVB"]
# x <- SD$Adjusted
# plot(x,type="line")
# MACD(x)
# RSI(x)


data(ttrc)
plot(ttrc[["Close"]])
dmi.adx <- ADX(ttrc[,c("High","Low","Close")])
# plot(dmi.adx)

temp <- as.data.table(cbind(Date = ttrc[["Date"]],close = ttrc[["Close"]],dmi.adx))
md <- melt(temp[Date>9500 & Date <10000],id.vars = "Date")
ggplot(md,aes(x=Date,y=value,colour=variable))+
  geom_line()