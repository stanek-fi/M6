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
  Stock[,PriceRatio := c(NA,Stock$Adjusted[-1]/Stock$Adjusted[-nrow(Stock)])]
  Stock[,StockName := StockName]
  # Stock[,.(Return = last(Adjusted)/first(Adjusted) - 1, StockName = StockName), Interval]
}))


PriceRatios <- as.data.table(dcast(StocksAggr, index ~ StockName, value.var = "PriceRatio"))
# PriceRatios[10000:10100,1:10]

Tobs <- 100
N <- 10
x <- PriceRatios[(nrow(PriceRatios)-Tobs+1):nrow(PriceRatios),2:(N+1)]
x <- torch_tensor(as.matrix(x), dtype = torch_float())

normalizeWeights <- function(x){
  # x / torch_sum(torch_abs(x)) * 0.25
  x / torch_sum(torch_abs(x)) * 1
}

net = nn_module(
  initialize = function(N) {
    self$lin1 <- nn_linear(1, N, bias = F)
  },
  forward = function(x) {
    # RET <- torch_matmul(x - 1, self$lin1$parameters$weight / torch_sum(torch_abs(self$lin1$parameters$weight)))
    RET <- torch_matmul(x - 1, normalizeWeights(self$lin1$parameters$weight))
    ret <- torch_log(1 + RET)
    sret <- torch_sum(ret)
    Tobs <- ret$size()[1]
    sret / (1/(Tobs - 1) * torch_sum((ret - sret/Tobs)^2))
  }
)

model <- net(N=N)
model(x)

# temp <- model$parameters$lin1.weight
# 
# RET1 <- torch_matmul(x, temp / torch_sum(torch_abs(temp)))
# ret <- torch_log(RET1)
# torch_sum(ret)

criterion = function(sharp){
  -sharp
}
optimizer = optim_adam(model$parameters, lr = 0.01)


epochs = 1000
for(i in 1:epochs){
  optimizer$zero_grad()
  sharp = model(x)
  loss = criterion(sharp)
  loss$backward()
  optimizer$step()
  
  # Check Training
  if(i %% 100 == 0){
  # if(T){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}
temp <- model$parameters$lin1.weight
cbind(as_array(normalizeWeights(temp)),apply(as.matrix(x),2,mean))


temp <- model$parameters$lin1.weight
RET <- torch_matmul(x - 1, temp / torch_sum(torch_abs(temp)))
ret <- torch_log(1 + RET)
sret <- torch_sum(ret)
Tobs <- ret$size()[1]
sret / (1/(Tobs - 1) * torch_sum((ret - sret/Tobs)^2))


