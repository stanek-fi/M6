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
QuantilePrediction <- QuantilePredictions$meta


submission <- QuantilePrediction[Split=="Validation" & M6Dataset == 1]
period <- submission[,str_c(unique(IntervalStart), " - " ,unique(IntervalEnd))]
submission <- submission[match(template$ID, submission$Ticker), .(ID = Ticker, Rank1 , Rank2, Rank3, Rank4, Rank5, Decision = 0)]
submission[, Decision := 0.01 * 0.25]

submission <- validateSubmission(submission, Round = T)

write.csv(submission,file.path("Results",str_c("Submission_", period, ".csv")),row.names = F, quote=FALSE)




