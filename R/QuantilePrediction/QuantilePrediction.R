library(data.table)
library(stringr)
library(torch)
rm(list=ls())
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")

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


# 
SD <- Stocks[Ticker == "AVB"]
lags = 0:2

LagReturn <- function(SD,lags = 1) {
  temp <- SD[,.(Return = last(Adjusted)/first(Adjusted) - 1), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Return, n=lags, fill = 0)]))
  names(temp) = c("Interval", str_c("ReturnLag",lags))
  return(temp)
}
LagVolatility <- function(SD,lags = 1) {
  temp <- SD[,.(Volatility = mean(diff(log(Adjusted))^2)), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Volatility, n=lags, fill = 0)]))
  names(temp) = c("Interval", str_c("VolatilityLag",lags))
  return(temp)
}

StocksAggr <- Stocks[,c(LagVolatility(.SD,0:2), LagReturn(.SD,0:2)),.(Ticker)]
# all(StocksAggr[,2] ==StocksAggr[,6])
StocksAggr[[6]]=NULL
featureNames <- names(StocksAggr)[-(1:2)]
apply(StocksAggr,2,function(x) mean(is.na(x)))
StocksAggr <- StocksAggr[,lapply(.SD, function(x) imputeNA(x))]
apply(StocksAggr,2,function(x) mean(is.na(x)))
StocksAggr <- StocksAggr[,c(.(Ticker=Ticker, Return = ReturnLag0),lapply(.SD, function(x) standartize(x))), Interval, .SDcols = featureNames]

# StocksAggr <- StocksAggr[,c(.(Ticker=Ticker, Return = ReturnLag0),lapply(.SD, function(x) x)), Interval, .SDcols = c("ReturnLag0", "ReturnLag1", "ReturnLag2")]
StocksAggr[,ReturnQuintile := computeQuintile(Return), Interval]
StocksAggr[which(apply(is.na(StocksAggr),1,any))]

# StocksAggr[,sd(ReturnLag0),Interval]

y <- StocksAggr[,ReturnQuintile]
# x <- StocksAggr[,.SD,.SDcols = c("VolatilityLag0", "VolatilityLag1", "VolatilityLag2", "ReturnLag0", "ReturnLag1", "ReturnLag2")]
x <- StocksAggr[,.SD,.SDcols = c("VolatilityLag0", "VolatilityLag1", "VolatilityLag2", "ReturnLag1", "ReturnLag2")]
# x <- StocksAggr[,.SD,.SDcols = c("ReturnLag1", "ReturnLag2")]

x_train = torch_tensor(as.matrix(x), dtype = torch_float())
y_train = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())

Model = nn_module(
  initialize = function() {
    self$lin1 <- nn_linear(5, 8)
    self$lin2 <- nn_linear(8, 16)
    self$lin3 <- nn_linear(16, 5)
  },
  forward = function(x) {
    x <- nnf_relu(self$lin1(x))
    x <- nnf_relu(self$lin2(x))
    x <- nnf_softmax(self$lin3(x),2)
    x
  }
)
model <- Model()


criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}
optimizer = optim_adam(model$parameters, lr = 0.01)
epochs = 200

for(i in 1:epochs){
  optimizer$zero_grad()
  
  y_pred = model(x_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}

y_pred = model(x_train)
ComputeRPSTensor(y_pred,y_train)
ComputeRPSTensor(torch_tensor(matrix(0.2,ncol=5,nrow=length(y)), dtype = torch_float()),y_train)




