library(data.table)
library(stringr)
library(torch)
rm(list=ls())

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
# SD <- Stocks[Ticker == "XOM"]

LagReturn <- function(SD,lags = 1) {
  temp <- SD[,.(Return = last(Adjusted)/first(Adjusted) - 1), Interval]
  temp <- cbind(data.table(Interval = temp[,Interval]), as.data.table(temp[,shift(Return, n=lags)]))
  names(temp) = c("Interval", str_c("ReturnLag",lags))
  return(temp)
}

StocksAggr <- Stocks[,LagReturn(.SD,0:2),.(Ticker)]


computeQuintile <- function(x){
  findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
} 
StocksAggr[,ReturnQuintile := computeQuintile(ReturnLag0), Interval]


y <- StocksAggr[,ReturnQuintile]
x <- StocksAggr[,.SD, .SDcols = c("ReturnLag0", "ReturnLag1", "ReturnLag2")]
# x <- StocksAggr[,.SD, .SDcols = c("ReturnLag1", "ReturnLag2")]
x <- as.matrix(x)
x[is.na(x)]=0


x_train = torch_tensor(x, dtype = torch_float())
y_train = torch_tensor(y,dtype = torch_long())



model = nn_sequential(
  # Layer 1
  nn_linear(3, 8),
  nn_relu(),
  # Layer 2
  nn_linear(8, 16),
  nn_relu(),
  # Layer 3
  nn_linear(16,5)
)


# model = nn_sequential(
#   # Layer 1
#   nn_linear(3, 5),
# )
RPS_tensor <- function(y_pred,yd){
  temp <- (y_pred$cumsum(2) - yd)^2
  mean(temp$sum(2)/5)
}

# criterion = nn_cross_entropy_loss()
criterion = function(y_pred,yd) {RPS_tensor(y_pred,yd)}
yd <- torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())
optimizer = optim_adam(model$parameters, lr = 0.01)
epochs = 200


  
for(i in 1:epochs){
  optimizer$zero_grad()
  
  y_pred = model(x_train)
  # loss = criterion(y_pred, y_train)
  loss = criterion(nnf_softmax(y_pred,2), yd)
  loss$backward()
  optimizer$step()
  
  
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}

# y_pred = model(x_train)
# y_pred <- as.array(nnf_softmax(y_pred,2))
# cbind(round(y_pred,2),y)

y_pred = model(x_train)
y_pred <- nnf_softmax(y_pred,2)
yd <- torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())

RPS_tensor(y_pred,yd)
RPS_tensor(torch_tensor(matrix(0.2,ncol=5,nrow=length(y)), dtype = torch_float()),yd)







# StocksAggr[Interval=="2022-01-10 : 2022-02-06"][order(ReturnLag0)]




Stocks[,Return := last(Adjusted)/first(Adjusted) - 1, .(Ticker, Interval)]



computeQuintile <- function(x){
  findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
} 

StocksAggr[,table(Interval)]
StocksAggr[,ReturnQuintile := computeQuintile(Return), Interval]




s <- 1
Stock <- Stocks[[s]]
StockName <- names(Stocks)[s]
colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")             
Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
SD <- Stock

f <- function(SD) {
  SD$Adjusted
  SD$Interval
}




























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