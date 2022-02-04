library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls())
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")


template <- read.csv(file.path("Data","template.csv"))
StockNames <- readRDS(file.path("Data","StockNames.RDS"))
Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
QuantilePredictions <- readRDS(file.path("Precomputed","QuantilePredictions.RDS"))


# plot(Stocks[["VXX"]]$VXX.Adjusted)

QuantilePrediction <- QuantilePredictions$meta

# temp <- QuantilePrediction[Split == "Test"][order(Rank5 - Rank1)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# View(temp)
# temp <- QuantilePrediction[Split == "Test"][order(Rank5 + Rank4 - Rank1 - Rank2)][,c(.(Interval,Ticker), lapply(.SD, function(x) round(x,3))), .SDcols= c("Return", str_c("Rank", 1:5))]
# View(temp)
submission <- QuantilePrediction[Split=="Validation"]
submission <- submission[match(template$ID, out$Ticker), .(ID = Ticker, Rank1 , Rank2, Rank3, Rank4, Rank5, Decision = 0)]
# View(submission[order(Rank5 - Rank1)])
# View(submission[order(Rank5 + Rank4 - Rank1 - Rank2)])
submission[ID == "VXX", Decision := -.125]
submission[ID == "V", Decision := .0625]
submission[ID == "CDW", Decision := .0625]



submission <- validateSubmission(submission, NormProb = T)
# submission <- validateSubmission(submission)

validateSubmission <- function(submission, NormProb = F){
  if(NormProb){
    norm <- apply(submission[,2:6],1,sum)
    message(str_c("NormProb discrepancy:", max(abs(norm-1))))
    submission[,2:6] <- submission[,2:6]/norm
  }
  template <- read.csv(file.path("Data","template.csv"))
  ordering <- all(template$ID == submission$ID)
  columns <- all(colnames(template) == colnames(submission))
  probsSumToOne <- all(abs(apply(submission[,2:6],1,sum) - 1) < 1e-8)
  probs0 <- all(submission[,2:6] >= 0)
  probs1 <- all(submission[,2:6] <= 1)
  minSumDecision <- sum(abs(submission$Decision)) >= .25
  maxSumDecision <- sum(abs(submission$Decision)) <= 1
  validity <- c(ordering = ordering, columns = columns, probsSumToOne = probsSumToOne, probs0 = probs0, probs1 = probs1, minSumDecision = minSumDecision, maxSumDecision = maxSumDecision)
  if(!all(validity)){
    stop(str_c("Invalid Submission,", names(validity)[!validity]))
  }
  return(submission)
}



