library(quantmod)
library(data.table)
require(TTR)
# library(BatchGetSymbols)
library(stringr)
library(imputeTS)
set.seed(1)
source("R/GenData/GenData_Helpers.R")

StockNames <- readRDS(file.path("Data", "StockNames.RDS"))
StockNames[Symbol=="FB",Symbol := "META"]
StockNames[,MinDate := as.Date(NA)]
StockNames[,MaxDate := as.Date(NA)]
StockNames[,Activity := as.numeric(NA)]
StockNames[,missings := 0]
# tickers <- StockNames[, Symbol]
tickers <- StockNames[M6Dataset>0, Symbol]
  

# Downloading data --------------------------------------------------------

from = "1970-01-01"
to = "2023-12-01"
Stocks <- lapply(seq_along(tickers), function(i) {
  Sys.sleep(1)
  print(round(i/length(tickers),3))
  ticker <- tickers[i]
  out <- try({
    temp <- as.data.table(getSymbols(ticker,from = from,to = to,auto.assign=FALSE))
    colnames(temp) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
    StockNames[i, MinDate := min(temp$index)]
    StockNames[i, MaxDate := max(temp$index)]
    StockNames[i, Activity := temp[max(1, .N - 100):.N, mean(Volume * Close)]]
    temp
  })
  if("try-error" %in% class(out)){
    out <- NULL
  }
  return(out)
})
if(length(Stocks) == length(tickers)){
  names(Stocks)=tickers
}else{
  stop("shorter data")
}
print(sum(!sapply(Stocks, function(y) {!is.null(y)})))

if(sum(!(tickers %in% names(Stocks)))>0){
  warning(str_c(sum(!(tickers %in% names(Stocks))), " stock missing"))
}


Stocks <- Stocks[sapply(Stocks, function(y) {!is.null(y)})]
table(as.Date(sapply(Stocks, function(s) s[.N,index])))


# Cleaning data -----------------------------------------------------------

# plot(Stocks[["IEFM.L"]]$Adjusted) #manually correcting invalid data
# Stocks[["IEFM.L"]][Adjusted<100,Adjusted := NA] #not needed anymore

StocksClean<- setNames(lapply(names(Stocks), function(ticker) {
  stock <- Stocks[[ticker]]
  naRows <- sum(apply(is.na(stock),1,any))
  StockNames[Symbol == ticker, missings := naRows]
  if(naRows>0){
    print(str_c("Ticker: ",ticker, " Missing: ", naRows))
    return(cbind(stock[,.(index)],stock[,lapply(.SD, noisyInterpolation), .SDcols=names(stock)[-1]]))
  }else{
    return(stock)
  }
}),names(Stocks))

# plot(StocksClean[["IEFM.L"]]$Adjusted) 

saveRDS(StocksClean,file.path("Data","StocksAll.RDS"))
# StocksClean <- readRDS(file.path("Data","StocksAll.RDS"))



# # DatasetsGeneration ----------------------------------------------------------
# Volatilites <- sapply(StocksClean, function(Stock) {Stock[max((.N-500),1):.N, mean(diff(log(Adjusted))^2,na.rm=T)]})
# for(i in seq_along(Volatilites)){
#   StockNames[Symbol == names(Volatilites)[i], Volatility := Volatilites[i]]
# }
# # StockNames[,mean(Volatility),.(M6Dataset, ETF)][order(M6Dataset, ETF)]
# 
# StockNames[,M6Dataset := ifelse(!is.na(M6Id), 1, 0)]
# ActivityETF <- unlist(StockNames[M6Dataset == 1 & ETF == T, .(mu=mean(Activity, na.rm=T), sd=sd(Activity, na.rm=T))])
# ActivityStock <- unlist(StockNames[M6Dataset == 1 & ETF == F, .(mu=mean(Activity, na.rm=T), sd=sd(Activity, na.rm=T))])
# # StockNames[, M6Likelihood := ifelse(ETF == T, dnorm(Activity, ActivityETF["mu"], ActivityETF["sd"]), dnorm(Activity, ActivityStock["mu"], ActivityStock["sd"]))]  
# VolatilityETF <- unlist(StockNames[M6Dataset == 1 & ETF == T, .(mu=mean(Volatility, na.rm=T), sd=sd(Volatility, na.rm=T))])
# VolatilityStock <- unlist(StockNames[M6Dataset == 1 & ETF == F, .(mu=mean(Volatility, na.rm=T), sd=sd(Volatility, na.rm=T))])
# StockNames[, M6Likelihood := ifelse(ETF == T, dnorm(Volatility, VolatilityETF["mu"], VolatilityETF["sd"]), dnorm(Volatility, VolatilityStock["mu"], VolatilityStock["sd"]))]  
# 
# # StockNames[M6Dataset != 0,mean(is.na(M6Likelihood))]
# # summary(StockNames[M6Dataset != 0,M6Likelihood])
# 
# numDataset <- 10
# set.seed(1)
# MinimalMaxDate <- StockNames[,sort(unique(MaxDate),decreasing = T)][2]
# MaximalMinDate <- as.Date("2015-01-01")
# MinActivity <- StockNames[(M6Dataset == 1), min(Activity,na.rm = T)] *0.5
# temp <- StockNames[(ETF == F) & (M6Dataset != 1) & (missings < 50) & (MaxDate >= MinimalMaxDate) & (MinDate <= MaximalMinDate) & (Activity >= MinActivity)][order(M6Likelihood,decreasing = T)][1:((numDataset-1)*50)][, Symbol]
# StockNames[Symbol %in% temp, M6Dataset := sample(rep(2:numDataset,50))]
# temp <- StockNames[(ETF == T) & (M6Dataset != 1) & (missings < 50) & (MaxDate >= MinimalMaxDate) & (MinDate <= MaximalMinDate) & (Activity >= MinActivity)][order(M6Likelihood,decreasing = T)][1:((numDataset-1)*50)][, Symbol]
# StockNames[Symbol %in% temp, M6Dataset := sample(rep(2:numDataset,50))]
# 
# 
# # StockNames[,.N,.(M6Dataset,ETF)][order(M6Dataset,ETF)]
# # StockNames[M6Dataset==1,.(minMaxDate = min(MaxDate), maxMissings = max(missings))]
# 
# saveRDS(StockNames,file.path("Data","StockNames.RDS"))



