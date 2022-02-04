library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls())
tempFilePath <- "C:/Users/stane/M6temp"
# source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
# source("R/QuantilePrediction/QuantilePrediction_Features.R")
# source("R/QuantilePrediction/QuantilePrediction_Models.R")
# source("R/MetaModel/MetaModel.R")
source("R/PortfolioOptimizationNN/PortfolioOptimizationNN_Helpers.R")


template <- read.csv(file.path("Data","template.csv"))
StockNames <- readRDS(file.path("Data","StockNames.RDS"))
Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
QuantilePredictions <- readRDS(file.path("Precomputed","QuantilePredictions.RDS"))






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



















# plot(Stocks[["VXX"]]$VXX.Adjusted)

QuantilePrediction <- QuantilePredictions$meta

# temp <- QuantilePrediction[Split == "Test"][order(Rank5 - Rank1)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# View(temp)
# temp <- QuantilePrediction[Split == "Test"][order(Rank5 + Rank4 - Rank1 - Rank2)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# View(temp)
submission <- QuantilePrediction[Split=="Validation"]
submission <- submission[match(template$ID, submission$Ticker), .(ID = Ticker, Rank1 , Rank2, Rank3, Rank4, Rank5, Decision = 0)]
# View(submission[order(Rank5 - Rank1)])
# View(submission[order(Rank5 + Rank4 - Rank1 - Rank2)])
submission[ID == "VXX", Decision := -.125]
submission[ID == "V", Decision := .0625]
submission[ID == "CDW", Decision := .0625]


submission <- validateSubmission(submission, Round = T)

write.csv(submission,file.path("Results","SubmissionManual.csv"),row.names = F, quote=FALSE)




