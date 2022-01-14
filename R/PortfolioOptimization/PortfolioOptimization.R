library(data.table)
library(stringr)
library(torch)
library(tseries)
library(PerformanceAnalytics)
library(ggplot2)
library(tidyquant)
library(MASS)
rm(list=ls())
source("R/PortfolioOptimization/PortfolioOptimization_Helpers.R")
source("R/PortfolioOptimization/PortfolioOptimization_Models.R")
source("R/PortfolioOptimization/PortfolioOptimization_Copula.R")
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

# hist(pmin(na.omit(c(as.matrix(PriceRatios[,-(1:2)]))),2),breaks = 50)
# summary(na.omit(c(as.matrix(PriceRatios[,-(1:2)]))))
# quantile(na.omit(c(as.matrix(PriceRatios[,-(1:2)]))),probs = seq(0, 1, 0.05))
# sd(na.omit(c(as.matrix(PriceRatios[,-(1:2)]))))
mean(na.omit(c(as.matrix(PriceRatios[,-(1:2)]))))

auxiliaryColumns <- c("Time", "Interval")
set.seed(2)
N <- 5
StartDate <- as.Date("2000-01-10")
# StartDate <- as.Date("2015-01-18")
# StartDate <- as.Date("2020-01-13")
EndDate <-as.Date("2021-11-14")
subsetPriceRatios <- samplePriceRatios(PriceRatios, N, StartDate, EndDate, auxiliaryColumns)


models <- c(
  # createModelCombinations(laggedExPostOptimal, list(window=c(250,1000))),
  createModelCombinations(equalWeights),
  createModelCombinations(unfeasibleAnalyticAproximation),
  createModelCombinations(unfeasibleCopula, list(muInfo = c("known","mean","unknown"))),
  # unfeasibleAnalyticAproximation = unfeasibleAnalyticAproximation,
  createModelCombinations(unfeasibleExPostOptimal),
  createModelCombinations(unfeasibleBootstrap)
)

intervals <- subsetPriceRatios[,unique(Interval)]
res <- vector(mode = "list", length = length(intervals))
i <- 10
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
# benchmarkModel <- "unfeasibleExPostOptimal(numStarts = 1)"
benchmarkModel <- NULL
if(!is.null(benchmarkModel)){
  auxiliaryColumns <- "startDate"
  d <- cbind(d[,.SD,.SDcols=auxiliaryColumns], d[,.SD,.SDcols=colnames(d)[!(colnames(d) %in% auxiliaryColumns)]] - d[[benchmarkModel]])
}
md <- melt(d, id.vars="startDate", variable.name = "model")
md <- md[!str_detect(model,c("equal")) & !str_detect(model,c("Analytic"))]
ggplot(md,aes(x=startDate,y=value,colour=model))+
  # geom_line(alpha=0.2)+
  geom_ma(ma_fun = SMA, n = 4, linetype = "solid")
  # geom_ma(ma_fun = SMA, n = 4*4, linetype = "solid")









# 
# interval <- intervals[i]
# print(as.character(interval))
# startDate <- as.Date(word(interval,sep=":",1))
# endDate <- as.Date(word(interval,sep=":",2))
# tickers <- colnames(subsetPriceRatios)[!(colnames(subsetPriceRatios) %in% auxiliaryColumns)]
# xpast <- subsetPriceRatios[Time<startDate,.SD,.SDcols = tickers]
# x <- subsetPriceRatios[Interval == interval,.SD,.SDcols = tickers]
# 
# 
# R <- 2
# 
# TObs <- nrow(x)
# mu <- apply(x, 2, mean)
# xdm <- as.matrix(apply(x, 2, function(col) col - mean(col)))
# Sigma <- t(xdm) %*% xdm/ TObs 
# xSim <- simulateCopula(TObs = TObs, R = R, Sigma = Sigma, mu = mu)
# 























