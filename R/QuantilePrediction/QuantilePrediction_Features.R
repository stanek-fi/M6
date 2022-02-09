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


TTRWrapper <- function(SD, f, SDcols, Normalize, SDcolsOut = NULL, SDcolsPlus = NULL, Transform = NULL, ...){
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
  SDcolsOut <- colnames(temp)
  if(length(Normalize)==1){
    Normalize <- rep(Normalize, length(SDcolsOut))
  }
  for(i in seq_along(Normalize)){
    if(Normalize[i]){
      temp[,i] <- temp[,i] / SD$Close
    }
  }
  if(!is.null(Transform)){
    if(length(Transform)==1){
      Transform <- rep(Transform, length(SDcolsOut))
    }
    for(i in seq_along(Transform)){
      temp[,i] <- Transform[[i]](temp[,i]) 
    }
  }
  temp <- cbind(SD,temp)
  temp <- temp[,lapply(.SD, function(x) mean(x, na.rm = T)),Interval, .SDcols=SDcolsOut]
  temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=SDcolsOut]
}


TTR <- list(
  function(SD, BY) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = F)},
  function(SD, BY) {TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F)},
  function(SD, BY) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T)},
  function(SD, BY) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F))},
  function(SD, BY) {TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "cci")},
  function(SD, BY) {TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
  function(SD, BY) {TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"),Normalize = F, SDcolsOut = "chaikinVolatility")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")},
  function(SD, BY) {TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")},
  # function(SD, BY) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")}, #temporaly disabled, this one is slow 
  function(SD, BY) {TTRWrapper(SD = SD, f = DonchianChannel, SDcols = c("High", "Low"), Normalize = T)},
  # function(SD, BY) {TTRWrapper(SD = SD, f = DVI, SDcols = c("Adjusted"), Normalize = F, n = 100)} # n changed from 252 to account for the shortest series...still fails
  # function(SD, BY) {TTRWrapper(SD = SD, f = EMV, SDcols = c("High", "Low"), SDcolsPlus = "Volume", Normalize = T)} #fails for no reason
  function(SD, BY) {TTRWrapper(SD = SD, f = GMMA, SDcols = c("Close"), Normalize = T, short = c(10), long=c(30,60))},
  function(SD, BY) {TTRWrapper(SD = SD, f = KST, SDcols = c("Adjusted"), Normalize = F)},
  function(SD, BY) {TTRWrapper(SD = SD, f = MACD, SDcols = c("Adjusted"), Normalize = F)},
  function(SD, BY) {TTRWrapper(SD = SD, f = MFI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "MFI")},
  function(SD, BY) {TTRWrapper(SD = SD, f = OBV, SDcols = c("Close"), Normalize = F, SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
  function(SD, BY) {TTRWrapper(SD = SD, f = PBands, SDcols = c("Close"), Normalize = T)},
  function(SD, BY) {TTRWrapper(SD = SD, f = ROC, SDcols = c("Close"), Normalize = F, SDcolsOut = "ROC")},
  function(SD, BY) {TTRWrapper(SD = SD, f = RSI, SDcols = c("Close"), Normalize = F, SDcolsOut = "RSI")},
  function(SD, BY) {TTRWrapper(SD = SD, f = runPercentRank, SDcols = c("Close"), Normalize = F, SDcolsOut = "runPercentRank", n=100)},
  function(SD, BY) {TTRWrapper(SD = SD, f = SAR, SDcols =  c("High", "Low"), Normalize = T)},
  function(SD, BY) {TTRWrapper(SD = SD, f = EMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "EMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = DEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "DEMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = EVWMA, SDcols = c("Close"), SDcolsPlus = "Volume", Normalize = T, SDcolsOut = "EVWMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ZLEMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "ZLEMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = HMA, SDcols = c("Close"), Normalize = T, SDcolsOut = "HMA")},
  function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=20)},
  function(SD, BY) {TTRWrapper(SD = SD, f = SNR, SDcols = c("High", "Low", "Close"), Normalize = F, n=60)},
  # function(SD, BY) {TTRWrapper(SD = SD, f = stoch, SDcols = c("High", "Low", "Close"), Normalize = F)}, #error at some stock leading NA
  function(SD, BY) {TTRWrapper(SD = SD, f = SMI, SDcols = c("High", "Low", "Close"), Normalize = F)},
  function(SD, BY) {TTRWrapper(SD = SD, f = TDI, SDcols = c("Close"), Normalize = T)},
  function(SD, BY) {TTRWrapper(SD = SD, f = TRIX, SDcols = c("Adjusted"), Normalize = F)},
  function(SD, BY) {TTRWrapper(SD = SD, f = ultimateOscillator, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "ultimateOscillator")},
  function(SD, BY) {TTRWrapper(SD = SD, f = VHF, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "VHF")},
  function(SD, BY) {TTRWrapper(SD = SD, f = volatility, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "volatility")},
  function(SD, BY) {TTRWrapper(SD = SD, f = williamsAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "williamsAD", Transform = list(function(x) c(NA,diff(x))))},
  # function(SD, BY) {TTRWrapper(SD = SD, f = volatility, SDcols = c("Open", "High", "Low", "Close"), Normalize = F, SDcolsOut = "volatility", calc = "garman.klass")},
  function(SD, BY) {TTRWrapper(SD = SD, f = WPR, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "WPR")},
  function(SD, BY) {TTRWrapper(SD = SD, f = ZigZag, SDcols = c("High", "Low", "Close"), Normalize = T, SDcolsOut = "ZigZag")}
)

# 
# TTR_ADX <- function(SD) {
#   TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = F)
# }
# 
# TTR_aroon <- function(SD) {
#   TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F)
# }
# 
# TTR_ATR <- function(SD){
#   TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T)
# }
# 





