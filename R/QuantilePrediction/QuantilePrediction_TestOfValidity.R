
#run the preambule of QuantilePrediction


q <- 1
ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - q]
ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - q]



submission <- read.csv(file.path("Results",str_c("Submission_",ValidationStart," - ", ValidationEnd,".csv")))
submission



realization <- StocksAggr[IntervalStart == ValidationStart & IntervalEnd == ValidationEnd & M6Dataset == 1, .(Ticker, Return, ReturnQuintile)]
realization <- realization[match(submission$ID,realization$Ticker)]


y_pred <- torch_tensor(as.matrix(as.data.table(submission)[,.(Rank1, Rank2, Rank3, Rank4, Rank5)]))

y <- realization[,ReturnQuintile]
y <- torch_tensor(t(sapply(y,function(x) {
  if(is.na(x)){
    rep(NA,5)
  }else{
    replace(numeric(5), x:5, 1)
  }
})), dtype = torch_float())



ComputeRPSTensor(y_pred,y)