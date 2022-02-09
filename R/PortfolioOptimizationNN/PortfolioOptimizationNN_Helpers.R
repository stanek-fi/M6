round_preserve_sum <- function(x, digits = 0) {
  up <- 10 ^ digits
  x <-  x * up
  y <-  floor(x)
  indices <-  tail(order(x-y), round(sum(x)) - sum(y))
  y[indices] <-  y[indices] + 1
  y / up
}



validateSubmission <- function(submission, Round = F){
  if(Round){
    submissionOrig <- submission
    submission[,2:6] <- as.data.table(t(apply(submission[,2:6],1,function(x) {round_preserve_sum(x/sum(x), digits = 5)})))
    submission$Decision <- round_preserve_sum(submission$Decision, digits = 5)
    message(str_c("Max rounding diff: ", max(abs(submissionOrig[,2:7] - submission[,2:7]))))
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



GenReturnArray <- function(Stocks,IntervalInfos){
  s <- 1
  colnamesStock <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
  StocksLong <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
    if(s %% 10 == 0){
      print(str_c("Stock:", s, " Time:", Sys.time()))
    }
    StockLong <- lapply(IntervalInfos, function(IntervalInfo){
      Stock <- Stocks[[s]]
      Ticker <- names(Stocks)[s]
      colnames(Stock) <- colnamesStock
      Stock <- AugmentStock(Stock[index>=IntervalInfo$Start & index<=IntervalInfo$End], IntervalInfo$End)
      Stock[, Return := c(NA,Stock$Adjusted[-1]/Stock$Adjusted[-nrow(Stock)] - 1)]
      Stock[, Interval := findInterval(index,IntervalInfo$TimeBreaks,left.open=T)]
      Stock[, Interval := factor(Interval, levels = seq_along(IntervalInfo$IntervalNames), labels = IntervalInfo$IntervalNames)]
      Stock[, Ticker := Ticker]
      Stock[, Shift := IntervalInfo$Shift]
      Stock[, IntervalStart := as.Date(str_sub(Interval,1,10))]
      Stock[, IntervalEnd := as.Date(str_sub(Interval,14,23))]
      # Stock[, IntervalPosition := match(index,seq(first(IntervalStart),first(IntervalEnd),by=1)), Interval]
      Stock[, IntervalPosition := match(index,seq(first(IntervalStart),first(IntervalEnd),by=1)[!(weekdays(seq(first(IntervalStart),first(IntervalEnd),by=1),abbreviate=T) %in% c("so", "ne"))]), Interval]
      Stock
      # Stock[, IntervalPosition := 1:.N, Interval]
      # Stock[,length(unique(IntervalPosition)),Interval]
      # Stock[Interval == "2022-02-07 : 2022-03-06"]
    })
    do.call(rbind,StockLong)
  }))
  if(StocksLong[,any(is.na(IntervalPosition))]){
    stop()
  }
  temp <- dcast(StocksLong, Interval + IntervalStart + IntervalEnd ~ IntervalPosition + Ticker , value.var = "Return")
  QuantilePredictionIntervals <- temp[,.(Interval, IntervalStart, IntervalEnd)]
  names <- colnames(temp)[-(1:3)]
  temp <- lapply(1:(5*4), function(i) {
    SDcols <- names[word(names,sep = "_") == i]
    if(length(SDcols)>0){
      temp[, setNames(.SD, word(SDcols, sep="_", 2)), .SDcols = SDcols]
    }
  })
  if(length(unique(lapply(temp[sapply(temp,length)>0], function(x) colnames(x))))>1){
    stop()
  }
  ReturnArray <- abind(temp, along=3)
  dimnames(ReturnArray)[[1]] <- QuantilePredictionIntervals$Interval
  dimnames(ReturnArray)[[3]] <- which(sapply(temp,length)>0)
  ReturnArray
}


GenQuantilePredictionArray <- function(QuantilePrediction){
  temp <- lapply(1:5, function(i) {
    dcast(QuantilePrediction, Interval + IntervalStart + IntervalEnd + Shift + Split ~ Ticker, value.var = str_c("Rank",i))
  })
  if(length(unique(lapply(temp, function(x) colnames(x))))>1){
    stop()
  }
  if(length(unique(lapply(temp, function(x) x$Interval)))>1){
    stop()
  }
  QuantilePredictionIntervals <- temp[[1]][,.(Interval, IntervalStart, IntervalEnd, Shift, Split)]
  QuantilePredictionTickers <- colnames(temp[[1]])[-(1:5)]
  
  temp <- lapply(temp, function(x) as.matrix(x[,-(1:5)]))
  temp <- abind( temp, along=3 )
  dimnames(temp)[[1]] <- QuantilePredictionIntervals$Interval
  dimnames(temp)[[3]] <- str_c("Rank",1:5)
  QuantilePredictionArray <- temp
  
  temp <- dimnames(QuantilePredictionArray)[[1]]
  QuantilePredictionArray <- QuantilePredictionArray[order(temp),,]
  QuantilePredictionArray
}



ComputeSharpTensor <- function(weights, y, eps = 0) {
  RET <- torch_einsum("nkt,nk->nt",list(y,weights))
  ret <- torch_log(RET + 1)
  sret <- ret$sum(2)
  # sdp <- torch_std(ret,dim=2, unbiased=TRUE)
  sdp <- 1
  ((21*12) / sqrt(252)) * (1/20) * sret/ (sdp + eps)
}
