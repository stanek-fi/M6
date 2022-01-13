library(data.table)
library(stringr)
library(torch)
library(tseries)
library(PerformanceAnalytics)
library(ggplot2)
library(tidyquant)
rm(list=ls())
source("R/Helpers_PortfolioOptimization.R")
Stocks <- readRDS(file.path("Data","StocksSP500.RDS"))


TimeEnd <- as.Date("2023-01-08")
TimeStart <- TimeEnd - (7*4) * 1000
TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
TimeBreaksNames <- str_c(TimeBreaks[-length(TimeBreaks)]+1, ":" , TimeBreaks[-1])

StocksAggr <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
  Stock <- Stocks[[s]]
  Ticker <- names(Stocks)[s]
  colnames(Stock) <- c("Time", "Open", "High", "Low", "Close", "Volume", "Adjusted")             
  Stock[,Interval := findInterval(Time,TimeBreaks,left.open=T)]
  Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
  Stock[,PriceRatio := c(NA,Stock$Adjusted[-1]/Stock$Adjusted[-nrow(Stock)])]
  Stock[,Ticker := Ticker]
  # Stock[,.(Return = last(Adjusted)/first(Adjusted) - 1, StockName = StockName), Interval]
}))
PriceRatios <- as.data.table(dcast(StocksAggr, Time + Interval ~ Ticker, value.var = "PriceRatio"))



auxiliaryColumns <- c("Time", "Interval")
# set.seed(1)
N <- 100
StartDate <- as.Date("2000-01-10")
# StartDate <- as.Date("2015-01-18")
EndDate <-as.Date("2021-11-14")
subsetPriceRatios <- samplePriceRatios(PriceRatios, N, StartDate, EndDate, auxiliaryColumns)

models <- list(
  equalweights1 = function(xpast,x){
    temp <- optimizeSharp(x, alpha = 1, epochs = 0)
    list(
      weights = temp$weights,
      sharp = computeSharpMatrix(x,temp$weights)
    )
  },
  equalweights05 = function(xpast,x){
    temp <- optimizeSharp(x, alpha = 0.5, epochs = 0)
    list(
      weights = temp$weights,
      sharp = computeSharpMatrix(x,temp$weights)
    )
  },
  equalweights025 = function(xpast,x){
    temp <- optimizeSharp(x, alpha = 0.25, epochs = 0)
    list(
      weights = temp$weights,
      sharp = computeSharpMatrix(x,temp$weights)
    )
  },
  # unfeasibleExPostOptimal1 = function(xpast,x){
  #   temp <- optimizeSharp(x, alpha = 1, epochs = 100)
  #   list(
  #     weights = temp$weights,
  #     sharp = computeSharpMatrix(x,temp$weights)
  #   )
  # },
  # unfeasibleExPostOptimal025 = function(xpast,x){
  #   temp <- optimizeSharp(x, alpha = 0.25, epochs = 100)
  #   list(
  #     weights = temp$weights,
  #     sharp = computeSharpMatrix(x,temp$weights)
  #   )
  # }
  feasibleLaggedPostOptimalFull = function(xpast,x){
    if(nrow(xpast)>0){
      temp <- optimizeSharp(xpast, alpha = 0.25, epochs = 100)
      weights <- temp$weights
    }else{
      weights <- rep(1/N,N) * 0.25
    }
    list(
      weights = weights,
      sharp = computeSharpMatrix(x,weights)
    )
  },
  feasibleLaggedPostOptimalRolling250 = function(xpast,x){
    window <- 250
    if(nrow(xpast)>window){
      temp <- optimizeSharp(xpast[(nrow(xpast)-window+1):nrow(xpast),], alpha = 0.25, epochs = 100)
      weights <- temp$weights
    }else{
      weights <- rep(1/N,N) * 0.25
    }
    list(
      weights = weights,
      sharp = computeSharpMatrix(x,weights)
    )
  },
  feasibleLaggedPostOptimalRolling1000 = function(xpast,x){
    window <- 1000
    if(nrow(xpast)>window){
      temp <- optimizeSharp(xpast[(nrow(xpast)-window+1):nrow(xpast),], alpha = 0.25, epochs = 100)
      weights <- temp$weights
    }else{
      weights <- rep(1/N,N) * 0.25
    }
    list(
      weights = weights,
      sharp = computeSharpMatrix(x,weights)
    )
  }
)

tickers <- colnames(subsetPriceRatios)[!(colnames(subsetPriceRatios) %in% auxiliaryColumns)]
x <- subsetPriceRatios[,.SD,.SDcols = tickers]
# temp <- lapply(models, function(model) model(xpast,x))
# sapply(temp, function(x) x$sharp)


intervals <- subsetPriceRatios[,unique(Interval)]
res <- vector(mode = "list", length = length(intervals))
i <- 1
for(i in seq_along(intervals)){
  interval <- intervals[i]
  print(as.character(interval))
  startDate <- as.Date(word(interval,sep=":",1))
  endDate <- as.Date(word(interval,sep=":",2))
  tickers <- colnames(subsetPriceRatios)[!(colnames(subsetPriceRatios) %in% auxiliaryColumns)]
  xpast <- subsetPriceRatios[Time<startDate,.SD,.SDcols = tickers]
  x <- subsetPriceRatios[Interval == interval,.SD,.SDcols = tickers]
  res[[i]] <- list(
    models = lapply(models, function(model) model(xpast,x)),
    info = list(
      interval = interval,
      startDate = startDate,
      endDate = endDate
    )
  )
}


d <- do.call(rbind,lapply(res, function(x) {cbind(startDate = x$info$startDate, as.data.table(lapply(x$models,function(y) y$sharp)))}))
md <- melt(d, id.vars="startDate", variable.name = "model")


# mdsub <- md[model %in% c("equalweights1", "equalweights05", "equalweights025")]
mdsub <- md[model %in% c("equalweights025", "feasibleLaggedPostOptimalFull", "feasibleLaggedPostOptimalRolling250", "feasibleLaggedPostOptimalRolling1000")]
ggplot(mdsub,aes(x=startDate,y=value,colour=model))+
  # geom_line(alpha=0.2)+
  # geom_ma(ma_fun = SMA, n = 4, linetype = "solid")
  geom_ma(ma_fun = SMA, n = 4*4, linetype = "solid")
  # coord_cartesian(ylim=c(-100,200))
















x <- res[[1]]

cbind(startDate = x$info$startDate, as.data.table(lapply(x$models,function(y) y$sharp)))

data.table(startDate = x$info$startDate, sapply(x$models,function(y) y$sharp))
names(x)




startTime <- word(interval,sep=":",1)
endTime <- word(interval,sep=":",2)


xpast <- subsetPriceRatios[Time<startTime,.SD,.SDcols = tickers]
x <- subsetPriceRatios[Interval == interval,.SD,.SDcols = tickers]






temp <- optimizeSharp(x, alpha = 1, epochs = 0)
temp$sharp

temp <- optimizeSharp(x, alpha = .5, epochs = 0)
temp$sharp

temp <- optimizeSharp(x, alpha = 0.25, epochs = 0)
temp$sharp

temp <- optimizeSharp(x, alpha = 1)
temp$sharp

temp <- optimizeSharp(x, alpha = 0.25, start = rnorm(N), silent = F)
temp$sharp












sum(t(apply(timeSubset,2,function(x) mean(is.na(x))))==1)

row.sum(is.na(timeSubset))
sum(colMeans(is.na(timeSubset))==0)

Tobs <- 250
N <- 50
x <- PriceRatios[(nrow(PriceRatios)-Tobs+1):nrow(PriceRatios),3:(N+2)]
x <- torch_tensor(as.matrix(x), dtype = torch_float())


set.seed(1)
N <- 3



dataSampler <- function(PriceRatios){
  
}


t(apply(PriceRatios,2,function(y) mean(is.na(y))))



temp <- optimizeSharp(x, alpha = 1, epochs = 0)
# temp$weights
temp$sharp

temp <- optimizeSharp(x, alpha = .5, epochs = 0)
# temp$weights
temp$sharp

temp <- optimizeSharp(x, alpha = 0.25, epochs = 0)
# temp$weights
temp$sharp

temp <- optimizeSharp(x, alpha = 1)
# temp$weights
temp$sharp

temp <- optimizeSharp(x, alpha = 0.25)
# temp$weights
temp$sharp








data(EuStockMarkets)
dax <- EuStockMarkets[,"DAX"]
ftse <- EuStockMarkets[,"FTSE"]
sharpe(log(dax))
sharpe(dax)
sharpe(log(ftse))
sharpe(ftse)


mean(diff(log(dax))) / sqrt(var(diff(log(dax))))


data(managers)
SharpeRatio.annualized(managers[,1,drop=FALSE], Rf=.035/12) 
SharpeRatio.annualized(managers[,1,drop=FALSE], Rf = managers[,10,drop=FALSE])
SharpeRatio.annualized(managers[,1:6], Rf=.035/12) 
SharpeRatio.annualized(managers[,1:6], Rf = managers[,10,drop=FALSE])
SharpeRatio.annualized(managers[,1:6], Rf = managers[,10,drop=FALSE],geometric=FALSE)


SharpeRatio.annualized(managers[,1,drop=FALSE], Rf=0) 
SharpeRatio.annualized(managers[,1], Rf=0) 
SharpeRatio(managers[,1], Rf=0)
temp <- managers[,1]
mean(temp) / sqrt(var(temp))









temp <- cbind(dax,ftse)
SharpeRatio(R = temp[,1,drop=FALSE])


net = nn_module(
  initialize = function(N,alpha) {
    # self$lin1 <- nn_linear(1, N, bias = F)
    self$weights = nn_parameter(normalizeWeights(torch_ones(N,1), alpha))
    self$alpha = alpha
  },
  forward = function(x) {
    weights <- normalizeWeights(self$weights, self$alpha)
    computeSharp(x,weights)
  }
)
model <- net(N=N, alpha = 1)
model(x)
model <- net(N=N, alpha = .25)
model(x)


model$parameters

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
# temp <- model$parameters$weights
# cbind(as_array(normalizeWeights(temp,model$alpha)),apply(as.matrix(x),2,mean))






temp <- model$parameters$lin1.weight
RET <- torch_matmul(x - 1, temp / torch_sum(torch_abs(temp)))
ret <- torch_log(1 + RET)
sret <- torch_sum(ret)
Tobs <- ret$size()[1]
sret / (1/(Tobs - 1) * torch_sum((ret - sret/Tobs)^2))


