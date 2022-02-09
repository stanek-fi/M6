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


s <- 1
IntervalInfo <- IntervalInfos[[1]]
colnamesStock <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
Stock <- Stocks[[s]]
Ticker <- names(Stocks)[s]
colnames(Stock) <- colnamesStock
Stock <- AugmentStock(Stock[index>=IntervalInfo$Start & index<=IntervalInfo$End], IntervalInfo$End)
Stock[,Interval := findInterval(index,IntervalInfo$TimeBreaks,left.open=T)]
Stock[,Interval := factor(Interval, levels = seq_along(IntervalInfo$IntervalNames), labels = IntervalInfo$IntervalNames)]
Stock[,Ticker := Ticker]
SD <- Stock

# SD <- Stocks[Ticker == "AVB"]

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


# temp1 <- TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
# temp1
# # temp2 <- TTRWrapper(SD = SD[index<as.Date("2022-01-10")], f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
# temp2 <- TTRWrapper(SD = SD[index<as.Date("2021-12-13")], f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
# temp2
# 
# 
# temp <- TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")
# temp[,time:= as.Date(str_sub(Interval, 1, 10))]
# ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
#   geom_line()



temp <- TTRWrapper(SD = SD, f = DonchianChannel, SDcols = c("High", "Low"), Normalize = T)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = DPO, SDcols = c("Adjusted"), Normalize = T, SDcolsOut = "DPO")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = DVI, SDcols = c("Adjusted"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = EMV, SDcols = c("High", "Low"), SDcolsPlus = "Volume", Normalize = T)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(10), long=c(30,60))
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = MACD, SDcols = c("Close"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = MFI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "MFI")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = OBV, SDcols = c("Close"), Normalize = F, SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = PBands, SDcols = c("Close"), Normalize = T)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = ROC, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = momentum, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = runPercentRank, SDcols = c("Close"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = SAR, SDcols =  c("High", "Low"), Normalize = T)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = SMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "SMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = EMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "EMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = WMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "WMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = EVWMA, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = ZLEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "WMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = VWAP, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = HMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "WMA")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()





temp <- TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=60)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()





temp <- TTRWrapper(SD = SD, f = stoch, SDcols = c("High", "Low", "Close"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = TDI, SDcols = c("Close"), Normalize = T)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F)
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = ultimateOscillator, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "ultimateOscillator")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = VHF, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "VHF")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()


temp <- TTRWrapper(SD = SD, f = volatility, SDcols = c("Open", "High", "Low", "Close"), Normalize = F, SDcolsOut = "volatility", calc = "garman.klass")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()



temp <- TTRWrapper(SD = SD, f = williamsAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "williamsAD", Transform = list(function(x) c(NA,diff(x))))
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()



temp <- TTRWrapper(SD = SD, f = WPR, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "WPR")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

temp <- TTRWrapper(SD = SD, f = ZigZag, SDcols = c("High", "Low", "Close"), Normalize = T, SDcolsOut = "ZigZag")
temp[,time:= as.Date(str_sub(Interval, 1, 10))]
ggplot(melt(temp, id.vars = c("Interval", "time")), aes(x=time, y=value, colour = variable))+
  geom_line()

