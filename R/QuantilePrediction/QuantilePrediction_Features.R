computeFeatures <- function(SD, BY, featureList){
  temp <- lapply(featureList, function(f) f(SD,BY))
  control <- lapply(temp, function(x) x[,Interval])
  if(length(unique.default(control))>1){
    stop("Some feature does not align.")
  }
  cbind(Interval = control[[1]], do.call(cbind,lapply(temp, function(x) x[,-1])))
}

# SD <- Stocks[Ticker == "ABBV"]
# SD <- Stock

firstNum <- function(x){
  temp <- na.omit(x)
  if(length(temp)>0){
    first(temp)
  }else{
    NA
  }
}

lastNum <- function(x){
  temp <- na.omit(x)
  if(length(temp)>0){
    last(temp)
  }else{
    NA
  }
}

Return <- function(SD) {
  temp <- SD[,.(Return = lastNum(Adjusted)/firstNum(Adjusted) - 1), Interval]
  return(temp)
}

IsETF <- function(SD, BY, StockNames) {
  ETF <- as.numeric(StockNames[Symbol == BY$Ticker, ETF])
  temp <- SD[,.(ETF = ETF), Interval]
  return(temp)
}

LagReturn <- function(SD, lags = 1) {
  temp <- SD[,.(Return = lastNum(Adjusted)/firstNum(Adjusted) - 1), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Return, n=lags, fill = NA)]))
  names(temp) = c("Interval", str_c("ReturnLag",lags))
  return(temp)
}

LagVolatility <- function(SD, lags = 1) {
  temp <- SD[,.(Volatility = mean(diff(log(na.omit(Adjusted)))^2)), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Volatility, n=lags, fill = NA)]))
  names(temp) = c("Interval", str_c("VolatilityLag",lags))
  return(temp)
}

Return <- function(SD) {
  temp <- SD[,.(Return = lastNum(Adjusted)/firstNum(Adjusted) - 1), Interval]
  return(temp)
}


TTRWrapper <- function(SD, f, SDcols, Normalize, NormalizeWith = NULL, SDcolsOut = NULL, SDcolsPlus = NULL, Transform = NULL, Suffix = NULL, Prefix = NULL, Aggreagation = "mean", ...){
# TTRWrapper <- function(SD, f, SDcols, Normalize, SDcolsOut = NULL, SDcolsPlus = NULL, Transform = NULL, Suffix = NULL, ...){
  # naRowsOld <- apply(is.na(SD[,.SD,.SDcols=SDcols]),1,any)
  naRows <- SD[,!complete.cases(.SD),.SDcols=SDcols]
  # if(!identical(naRowsOld,naRows)){
  #   stop("sdf")
  # }
  if(is.null(SDcolsPlus)){
    temp <- f(SD[!naRows,.SD,.SDcols=SDcols], ...)
  }else{
    temp <- f(SD[!naRows,.SD,.SDcols=SDcols], SD[!naRows,.SD,.SDcols=SDcolsPlus], ...)
  }
  if(is.vector(temp)){
    temp <- as.data.frame(temp)
    if(!is.null(SDcolsOut)){
      colnames(temp)=SDcolsOut
    }
  }
  temp <- temp[c(rep(NA, sum(!naRows)-nrow(temp)),1:nrow(temp)),,drop=F] #added to correct when product of f is of shorter lenght
  temp <- temp[ifelse(naRows,NA,cumsum(!naRows)),,drop=F]

  if(length(Normalize)==1){
    # Normalize <- rep(Normalize, length(SDcolsOut))
    Normalize <- rep(Normalize, ncol(temp))
  }
  if(is.null(NormalizeWith)){
    normcol <- SD$Close
  }else{
    normcol <- temp[,NormalizeWith]
  }
  for(i in seq_along(Normalize)){
    if(Normalize[i]==1){
      temp[,i] <- temp[,i] / normcol
    }else if (Normalize[i]==-1){
      temp[,i] <- temp[,i] - normcol
    }
  }
  if(!is.null(NormalizeWith)){
    temp <- temp[,colnames(temp) != NormalizeWith,drop=F]
  }
  
  if(!is.null(Transform)){
    if(length(Transform)==1){
      Transform <- rep(Transform, length(SDcolsOut))
    }
    for(i in seq_along(Transform)){
      temp[,i] <- Transform[[i]](temp[,i]) 
    }
  }
  SDcolsOut <- str_c(Prefix, colnames(temp), Suffix)
  colnames(temp) <- SDcolsOut
  temp <- cbind(SD,temp)
  
  if(Aggreagation == "mean"){
    temp <- temp[,lapply(.SD, function(x) mean(x, na.rm = T)),Interval, .SDcols=SDcolsOut]
  }else if(Aggreagation == "last"){
    temp <- temp[,lapply(.SD, function(x) last(x)), Interval, .SDcols=SDcolsOut]
  }else{
    stop()
  }
  temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=SDcolsOut]
  return(temp)
}


# TTR <- list(
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = F)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=7", n=7)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=14")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28", n=28)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28_last", n=28, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=32", n=32)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=32_last", n=32, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=46", n=46)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=46_last", n=46, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=100", n=100)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=100_last", n=100, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F))},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "cci")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"),Normalize = F, SDcolsOut = "chaikinVolatility")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")}, #temporaly disabled, this one is slow
#   function(SD, BY) {TTRWrapper(SD = SD, f = DonchianChannel, SDcols = c("High", "Low"), Normalize = T)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = DVI, SDcols = c("Adjusted"), Normalize = F, n = 100)} # n changed from 252 to account for the shortest series...still fails
#   # function(SD, BY) {TTRWrapper(SD = SD, f = EMV, SDcols = c("High", "Low"), SDcolsPlus = "Volume", Normalize = T)} #fails for no reason
#   function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(10), long=c(30,60))},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = F)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = F)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = MFI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "MFI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = OBV, SDcols = c("Close"), Normalize = F, SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = PBands, SDcols = c("Close"), Normalize = T)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ROC, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = runPercentRank, SDcols = c("Close"), Normalize = F, SDcolsOut = "runPercentRank", n=100)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SAR, SDcols =  c("High", "Low"), Normalize = T)},  # first 4 values are normlized accoring to the whole series. This leads to leakage flag...althout it should not be of pratical importance, Include it if you want.
#   function(SD, BY) {TTRWrapper(SD = SD, f = EMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "EMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=10", n=10)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=20", n=20)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=40", n=40)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = EVWMA, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ZLEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "ZLEMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = HMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "HMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=20)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=60)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = stoch, SDcols = c("High", "Low", "Close"), Normalize = F)}, #error at some stock leading NA
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=13", n=13)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=30", n=30)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=60", n=60)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TDI, SDcols = c("Close"), Normalize = T)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=10", n=10)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=20", n=20)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=40", n=40)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ultimateOscillator, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "ultimateOscillator")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = VHF, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "VHF")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=10", n=10)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=10_last", n=10, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=40", n=40)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=40_last", n=40, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=100", n=100)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=100_last", n=100, Aggreagation = "last")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Open", "High", "Low", "Close"), Normalize = F, SDcolsOut = "volatility", calc = "garman.klass", Suffix = "_(garman.klass)")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = williamsAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "williamsAD", Transform = list(function(x) c(NA,diff(x))))}
#   # function(SD, BY) {TTRWrapper(SD = SD, f = WPR, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "WPR")}
# )


# TTR <- list(
#   function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = T, NormalizeWith = "ADX", Prefix = "ADX1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = -1, NormalizeWith = "ADX", Prefix = "ADX2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F, Prefix = "aroon_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=7", Prefix = "ATR_", n=7)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=14", Prefix = "ATR_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28", Prefix = "ATR_", n=28)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28_last", Prefix = "ATR_", n=28, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=32", Prefix = "ATR_", n=32)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=32_last", Prefix = "ATR_", n=32, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=46", Prefix = "ATR_", n=46)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=46_last", Prefix = "ATR_", n=46, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=100", Prefix = "ATR_", n=100)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=100_last", Prefix = "ATR_", n=100, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F), Prefix = "BBands1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F), NormalizeWith = "mavg", Prefix = "BBands2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "cci", Prefix = "CCI_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"),Normalize = F, SDcolsOut = "chaikinVolatility")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")}, #temporaly disabled, this one is slow
#   function(SD, BY) {TTRWrapper(SD = SD, f = DonchianChannel, SDcols = c("High", "Low"), Normalize = T, Prefix = "DonchianChannel_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = DVI, SDcols = c("Adjusted"), Normalize = F, n = 100)} # n changed from 252 to account for the shortest series...still fails
#   # function(SD, BY) {TTRWrapper(SD = SD, f = EMV, SDcols = c("High", "Low"), SDcolsPlus = "Volume", Normalize = T)} #fails for no reason
#   function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(10), long=c(30,60), NormalizeWith = "short lag 10", Prefix = "GMMA1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(5,15), long=c(40), NormalizeWith = "long lag 40", Prefix = "GMMA2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = F, Prefix = "KST1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = -1, NormalizeWith = "signal", Prefix = "KST2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = F, Prefix = "MACD1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = -1, NormalizeWith = "macd", Prefix = "MACD2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = MFI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "MFI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = OBV, SDcols = c("Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "OBV",Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = PBands, SDcols = c("Close"), Normalize = T, Prefix = "PBands_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ROC, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = runPercentRank, SDcols = c("Close"), Normalize = F, SDcolsOut = "runPercentRank", n=100)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SAR, SDcols =  c("High", "Low"), Normalize = T)},  # first 4 values are normlized accoring to the whole series. This leads to leakage flag...althout it should not be of pratical importance, Include it if you want.
#   function(SD, BY) {TTRWrapper(SD = SD, f = EMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "EMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=10", n=10)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=20", n=20)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=40", n=40)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = EVWMA, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ZLEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "ZLEMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = HMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "HMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=20, SDcolsOut = "SNR", Suffix = "_n=20")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=60, SDcolsOut = "SNR", Suffix = "_n=20")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = stoch, SDcols = c("High", "Low", "Close"), Normalize = F)}, #error at some stock leading NA
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=13", n=13, Prefix = "SMI1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=30", n=30, Prefix = "SMI1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = -1, Suffix = "_n=13", n=13, Prefix = "SMI2_", NormalizeWith = "SMI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = -1, Suffix = "_n=30", n=30, Prefix = "SMI2_", NormalizeWith = "SMI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TDI, SDcols = c("Close"), Normalize = T, Prefix = "TDI_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=10", n=10, Prefix = "TRIX1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=20", n=20, Prefix = "TRIX1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=40", n=40, Prefix = "TRIX1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = -1, Suffix = "_n=10", n=10, Prefix = "TRIX2_", NormalizeWith = "TRIX")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = -1, Suffix = "_n=20", n=20, Prefix = "TRIX2_", NormalizeWith = "TRIX")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = -1, Suffix = "_n=40", n=40, Prefix = "TRIX2_", NormalizeWith = "TRIX")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ultimateOscillator, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "ultimateOscillator")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = VHF, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "VHF")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=10", n=10)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=10_last", n=10, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=40", n=40)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=40_last", n=40, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=100", n=100)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=100_last", n=100, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Open", "High", "Low", "Close"), Normalize = F, SDcolsOut = "volatility", calc = "garman.klass", Suffix = "_(garman.klass)")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = williamsAD, SDcols = c("High", "Low", "Close"), Normalize = T, SDcolsOut = "williamsAD", Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = WPR, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "WPR")}
# )



TTR <- list(
  function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = F, Prefix = "ADX1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F, Prefix = "aroon_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=14", Prefix = "ATR_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28", n=28, Prefix = "ATR_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=7", n=7, Prefix = "ATR_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F), Prefix = "BBands1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "cci")},
  function(SD, BY) {TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
  function(SD, BY) {TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"),Normalize = F, SDcolsOut = "chaikinVolatility")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")}, #temporaly disabled, this one is slow
  function(SD, BY) {TTRWrapper(SD = SD, f = DonchianChannel, SDcols = c("High", "Low"), Normalize = T, Prefix = "DonchianChannel_")},
  # function(SD, BY) {TTRWrapper(SD = SD, f = DVI, SDcols = c("Adjusted"), Normalize = F, n = 100)} # n changed from 252 to account for the shortest series...still fails
  # function(SD, BY) {TTRWrapper(SD = SD, f = EMV, SDcols = c("High", "Low"), SDcolsPlus = "Volume", Normalize = T)} #fails for no reason
  function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(10), long=c(30,60), Prefix = "GMMA1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = F, Prefix = "KST1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = F, Prefix = "MACD1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = MFI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "MFI")},
  function(SD, BY) {TTRWrapper(SD = SD, f = OBV, SDcols = c("Close"), Normalize = F, SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))), SDcolsOut = "OBV")},
  function(SD, BY) {TTRWrapper(SD = SD, f = PBands, SDcols = c("Close"), Normalize = T, Prefix = "PBands_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ROC, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")},
  function(SD, BY) {TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI")},
  function(SD, BY) {TTRWrapper(SD = SD, f = runPercentRank, SDcols = c("Close"), Normalize = F, SDcolsOut = "runPercentRank", n=100)},
  # function(SD, BY) {TTRWrapper(SD = SD, f = SAR, SDcols =  c("High", "Low"), Normalize = T)},  # first 4 values are normlized accoring to the whole series. This leads to leakage flag...althout it should not be of pratical importance, Include it if you want.
  function(SD, BY) {TTRWrapper(SD = SD, f = EMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "EMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = EVWMA, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ZLEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "ZLEMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = HMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "HMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=20, SDcolsOut = "SNR", Suffix = "_n=20")},
  function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=60, SDcolsOut = "SNR", Suffix = "_n=20")},
  # function(SD, BY) {TTRWrapper(SD = SD, f = stoch, SDcols = c("High", "Low", "Close"), Normalize = F)}, #error at some stock leading NA
  function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Prefix = "SMI1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = TDI, SDcols = c("Close"), Normalize = T, Prefix = "TDI_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Prefix = "TRIX1_")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ultimateOscillator, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "ultimateOscillator")},
  function(SD, BY) {TTRWrapper(SD = SD, f = VHF, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "VHF")},
  function(SD, BY) {TTRWrapper(SD = SD, f = volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility")},
  function(SD, BY) {TTRWrapper(SD = SD, f = williamsAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "williamsAD", Transform = list(function(x) c(NA,diff(x))))},
  function(SD, BY) {TTRWrapper(SD = SD, f = WPR, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "WPR")}
)



# TTR <- list(
#   function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = T, NormalizeWith = "ADX", Prefix = "ADX1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = -1, NormalizeWith = "ADX", Prefix = "ADX2_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F, Prefix = "aroon_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=7", Prefix = "ATR_", n=7)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=14", Prefix = "ATR_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28", Prefix = "ATR_", n=28)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=28_last", Prefix = "ATR_", n=28, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=32", Prefix = "ATR_", n=32)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=32_last", Prefix = "ATR_", n=32, Aggreagation = "last")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=46", Prefix = "ATR_", n=46)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=46_last", Prefix = "ATR_", n=46, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=100", Prefix = "ATR_", n=100)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T, Suffix = "_n=100_last", Prefix = "ATR_", n=100, Aggreagation = "last")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F), Prefix = "BBands1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F), NormalizeWith = "mavg", Prefix = "BBands2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "cci", Prefix = "CCI_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"),Normalize = F, SDcolsOut = "chaikinVolatility")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")}, #temporaly disabled, this one is slow
#   function(SD, BY) {TTRWrapper(SD = SD, f = DonchianChannel, SDcols = c("High", "Low"), Normalize = T, Prefix = "DonchianChannel_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = DVI, SDcols = c("Adjusted"), Normalize = F, n = 100)} # n changed from 252 to account for the shortest series...still fails
#   # function(SD, BY) {TTRWrapper(SD = SD, f = EMV, SDcols = c("High", "Low"), SDcolsPlus = "Volume", Normalize = T)} #fails for no reason
#   function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(10), long=c(30,60), NormalizeWith = "short lag 10", Prefix = "GMMA1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(5,15), long=c(40), NormalizeWith = "long lag 40", Prefix = "GMMA2_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = F, Prefix = "KST1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = -1, NormalizeWith = "signal", Prefix = "KST2_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = F, Prefix = "MACD1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = -1, NormalizeWith = "macd", Prefix = "MACD2_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = MFI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "MFI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = OBV, SDcols = c("Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "OBV",Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = PBands, SDcols = c("Close"), Normalize = T, Prefix = "PBands_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ROC, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI", Suffix = "_n=14", n=14)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI", Suffix = "_n=28", n=28)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = runPercentRank, SDcols = c("Close"), Normalize = F, SDcolsOut = "runPercentRank", n=100)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SAR, SDcols =  c("High", "Low"), Normalize = T)},  # first 4 values are normlized accoring to the whole series. This leads to leakage flag...althout it should not be of pratical importance, Include it if you want.
#   function(SD, BY) {TTRWrapper(SD = SD, f = EMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "EMA")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=10", n=10)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=20", n=20)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA", Suffix = "_n=40", n=40)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = EVWMA, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ZLEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "ZLEMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = HMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "HMA")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=20, SDcolsOut = "SNR", Suffix = "_n=20")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=60, SDcolsOut = "SNR", Suffix = "_n=20")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = stoch, SDcols = c("High", "Low", "Close"), Normalize = F)}, #error at some stock leading NA
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=13", n=13, Prefix = "SMI1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F, Suffix = "_n=30", n=30, Prefix = "SMI1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = -1, Suffix = "_n=13", n=13, Prefix = "SMI2_", NormalizeWith = "SMI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = -1, Suffix = "_n=30", n=30, Prefix = "SMI2_", NormalizeWith = "SMI")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TDI, SDcols = c("Close"), Normalize = T, Prefix = "TDI_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=10", n=10, Prefix = "TRIX1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=20", n=20, Prefix = "TRIX1_")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F, Suffix = "_n=40", n=40, Prefix = "TRIX1_")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = -1, Suffix = "_n=10", n=10, Prefix = "TRIX2_", NormalizeWith = "TRIX")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = -1, Suffix = "_n=20", n=20, Prefix = "TRIX2_", NormalizeWith = "TRIX")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = -1, Suffix = "_n=40", n=40, Prefix = "TRIX2_", NormalizeWith = "TRIX")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = ultimateOscillator, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "ultimateOscillator")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = VHF, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "VHF")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=10", n=10)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=10_last", n=10, Aggreagation = "last")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=40", n=40)},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=40_last", n=40, Aggreagation = "last")},
#   # function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=100", n=100)},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility", Suffix = "_n=100_last", n=100, Aggreagation = "last")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = TTR::volatility, SDcols = c("Open", "High", "Low", "Close"), Normalize = F, SDcolsOut = "volatility", calc = "garman.klass", Suffix = "_(garman.klass)")},
#   function(SD, BY) {TTRWrapper(SD = SD, f = williamsAD, SDcols = c("High", "Low", "Close"), Normalize = T, SDcolsOut = "williamsAD", Transform = list(function(x) c(NA,diff(x))))},
#   function(SD, BY) {TTRWrapper(SD = SD, f = WPR, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "WPR")}
# )




