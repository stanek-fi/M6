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



SD <- Stocks[Ticker == "AVB"]

#ADX
temp <- cbind(SD,ADX(SD[,.(High,Low,Close)]))
temp <- temp[,lapply(.SD, function(x) mean(x)),Interval, .SDcols=c("DIp", "DIn", "DX", "ADX")]
temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=c("DIp", "DIn", "DX", "ADX")]

# data(ttrc)
# plot(ttrc[["Close"]])
# dmi.adx <- ADX(ttrc[,c("High","Low","Close")],n=50)
# # plot(dmi.adx)
# 
# temp <- as.data.table(cbind(Date = ttrc[["Date"]],close = ttrc[["Close"]],dmi.adx))
# md <- melt(temp[Date>9800 & Date <10000],id.vars = "Date")
# ggplot(md,aes(x=Date,y=value,colour=variable))+
#   geom_line()


#aroon 
temp <- cbind(SD,aroon(SD[,.(High,Low)]))
temp <- temp[,lapply(.SD, function(x) mean(x)),Interval, .SDcols=c("aroonUp", "aroonDn", "oscillator")]
temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=c("aroonUp", "aroonDn", "oscillator")]

# data(ttrc)
# trend <- aroon( ttrc[,c("High", "Low")], n=20 )
# 
# temp <- as.data.table(cbind(Date = ttrc[["Date"]],close = ttrc[["Close"]],trend))
# md <- melt(temp[Date>9800 & Date <10000],id.vars = "Date")
# # md <- melt(temp,id.vars = "Date")
# ggplot(md,aes(x=Date,y=value,colour=variable))+
#   geom_line()


#atr 
# temp <- cbind(SD,ATR(SD[,.(High,Low, Close)]))
# 
# f <- ATR
# SDcols <- c("High", "Low", "Close")
# Normalize <- T
# 
# naRows <- apply(is.na(SD[,.SD,.SDcols=SDcols]),1,any)
# temp <- f(SD[!naRows,.SD,.SDcols=SDcols])
# temp <- temp[ifelse(naRows,NA,cumsum(!naRows)),]
# SDcols <- colnames(temp)
# if(Normalize){
#   temp <- apply(temp, 2, function(x) x / SD$Close)
# }
# temp <- cbind(SD,temp)
# temp <- temp[,lapply(.SD, function(x) mean(x, na.rm = T)),Interval, .SDcols=SDcols]
# temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=SDcols]

temp <- TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"),Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()




temp <- TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"),Normalize = c(T, T, T, F))
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


data(ttrc)
temp <- CCI(ttrc[,c("High","Low","Close")])
as.data.table(cci)


temp <- TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"),Normalize = F, SDcolsOut = "cci")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()



temp <- TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"),Normalize = F, SDcolsOut = "cci")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


#not included becouse it is of odd form and is not clear how should be normalized
data(ttrc)
ad <- chaikinAD(ttrc[,c("High","Low","Close")], ttrc[,"Volume"])

temp <- TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
# temp[,cci := c(NA, diff(cci))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()



data(ttrc)
volatility <- chaikinVolatility(ttrc[,c("High","Low")])

temp <- TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"), Normalize = F, SDcolsOut = "chaikinVolatility")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()



temp <- TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()



temp <- TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp1 <- TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
temp1
# temp2 <- TTRWrapper(SD = SD[index<as.Date("2022-01-10")], f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
temp2 <- TTRWrapper(SD = SD[index<as.Date("2021-12-13")], f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
temp2



