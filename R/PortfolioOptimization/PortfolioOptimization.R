library(data.table)
library(stringr)
library(torch)
library(tseries)
library(PerformanceAnalytics)
library(ggplot2)
library(tidyquant)
rm(list=ls())
source("R/PortfolioOptimization/PortfolioOptimization_Helpers.R")
source("R/PortfolioOptimization/PortfolioOptimization_Models.R")
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
set.seed(1)
N <- 100
# StartDate <- as.Date("2000-01-10")
StartDate <- as.Date("2015-01-18")
EndDate <-as.Date("2021-11-14")
subsetPriceRatios <- samplePriceRatios(PriceRatios, N, StartDate, EndDate, auxiliaryColumns)

models <- c(
  createModelCombinations(equalWeights, list(alpha=c(1,0.5,0.25))),
  createModelCombinations(unfeasibleExPostOptimal, list(alpha=c(1,0.25))),
  createModelCombinations(laggedExPostOptimal, list(window=c(250,1000)))
)

intervals <- subsetPriceRatios[,unique(Interval)]
res <- vector(mode = "list", length = length(intervals))
i <- 50
for(i in seq_along(intervals)){
  interval <- intervals[i]
  print(as.character(interval))
  startDate <- as.Date(word(interval,sep=":",1))
  endDate <- as.Date(word(interval,sep=":",2))
  tickers <- colnames(subsetPriceRatios)[!(colnames(subsetPriceRatios) %in% auxiliaryColumns)]
  xpast <- subsetPriceRatios[Time<startDate,.SD,.SDcols = tickers]
  x <- subsetPriceRatios[Interval == interval,.SD,.SDcols = tickers]
  res[[i]] <- list(
    models = lapply(models, function(model) 
      model(xpast,x)
    ),
    info = list(
      interval = interval,
      startDate = startDate,
      endDate = endDate
    )
  )
}


d <- do.call(rbind,lapply(res, function(x) {cbind(startDate = x$info$startDate, as.data.table(lapply(x$models,function(y) y$sharp)))}))
benchmarkModel <- "equalWeights(alpha = 1)"
if(!is.null(benchmarkModel)){
  auxiliaryColumns <- "startDate"
  d <- cbind(d[,.SD,.SDcols=auxiliaryColumns], d[,.SD,.SDcols=colnames(d)[!(colnames(d) %in% auxiliaryColumns)]] - d[[benchmarkModel]])
}

md <- melt(d, id.vars="startDate", variable.name = "model")
mdsub <- md[str_detect(model,c("equal"))]
# mdsub <- md
ggplot(mdsub,aes(x=startDate,y=value,colour=model))+
  # geom_line(alpha=0.2)+
  geom_ma(ma_fun = SMA, n = 4, linetype = "solid")
  # geom_ma(ma_fun = SMA, n = 4*4, linetype = "solid")
  # coord_cartesian(ylim=c(-100,200))







# 
# x <- array(1:30,dim=c(5,3,2))
# weights <- matrix(c(0.01,0.09,0.9))
# x[,,1]%*%weights
# x[,,2]%*%weights
# 
# xt <- torch_tensor(x, dtype = torch_float())
# xt$shape
# weightst <-  torch_tensor(weights,  dtype = torch_float())
# # torch_matmul(xt,weightst )
# 
# torch_einsum("mnp,nk->mkp",list(xt,weightst))
# torch_squeeze(torch_einsum("mnp,nk->mkp",list(xt,weightst)))
# 
# 
# N <- 2
# TObs <- 3
# R <- 4
# x <- array(rnorm(N*TObs*R,1,0.1),dim=c(TObs,N,R))
# weights <- as.matrix(rnorm(N))
# 
# N <- 2
# TObs <- 3
# R <- 1
# x <- array(rnorm(N*TObs*R,1,0.01),dim=c(TObs,N))
# weights <- as.matrix(rnorm(N))
# 
# 
# as.array(computeSharpTensor(torch_tensor(x[,,1]),torch_tensor(weights)))
# computeSharpMatrix(x[,,1],weights)
# as.array(computeSharpTensor(torch_tensor(x[,,2]),torch_tensor(weights)))
# computeSharpMatrix(x[,,2],weights)
# as.array(computeSharpTensor(torch_tensor(x[,,3]),torch_tensor(weights)))
# computeSharpMatrix(x[,,3],weights)
# 
# 
# as.array(computeSharpTensor(torch_tensor(x),torch_tensor(weights)))
# computeSharpMatrix(x,weights)
# 
# 
# # x <- torch_tensor(x, dtype = torch_float())
# if(length(dim(x))==3){
#   x <- torch_tensor(x, dtype = torch_float())
# }else{
#   x <- torch_tensor(array(x,dim=c(dim(x),1)), dtype = torch_float())
# }
# weights = torch_tensor(weights, dtype = torch_float())
# 
# RET <- torch_squeeze(torch_einsum("mnp,nk->mkp",list(x-1,weights)),dim=2)
# ret <- torch_log(1 + RET)
# sret <- torch_einsum("mn->n",ret)
# sdp <- torch_std(ret,dim=1, unbiased=TRUE)
# as.array(sret/sdp)
# 
# 
# alpha=1
# # model <- weightsModule(N=N,alpha=alpha)
# # model(x)
# optimizeSharp(x,alpha)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# RET <- torch_matmul(x - 1, weights)
# ret <- torch_log(1 + RET)
# sret <- torch_sum(ret)
# Tobs <- ret$size()[1]
# sdp <- 1/(Tobs - 1) * torch_sum((ret - sret/Tobs)^2)
# sret / sdp

