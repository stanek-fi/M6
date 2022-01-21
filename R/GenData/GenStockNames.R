library(quantmod)
library(data.table)
require(TTR)
library(BatchGetSymbols)

M6_Universe <- as.data.table(read.csv(file.path("Data","M6_Universe.csv")))
names(M6_Universe)[1] = "M6Id"
names(M6_Universe)[3] = "Symbol"
StockNames <- as.data.table(stockSymbols())
StockNames <- merge(StockNames, M6_Universe, by="Symbol", all = T)
StockNames[(!is.na(M6Id)) & class == "ETF", ETF := TRUE]
StockNames[(!is.na(M6Id)) & is.na(Name), Name := paste0(name, " (M6)")]
# StockNames[(!is.na(M6Id))][order(M6Id)]
StockNames$class <- NULL
StockNames$name <- NULL

SP500 <- GetSP500Stocks()
StockNames <- merge(StockNames, SP500, by.x="Symbol", by.y="Tickers",all.x=T)
# View(StockNames[(!is.na(M6Id))][order(M6Id)])
StockNames[,SP500 := (!is.na(Company)) | ((!is.na(M6Id)) & ETF == FALSE)]
StockNames[,Type := ifelse(!is.na(GICS_sector.ETF_type), GICS_sector.ETF_type, GICS.Sector)]
StockNames[,SubType := ifelse(!is.na(GICS_industry.ETF_subtype), GICS_industry.ETF_subtype, GICS.Sub.Industry)]
StockNames$CIK <- NULL
StockNames$GICS.Sector <- NULL
StockNames$GICS.Sub.Industry <- NULL
StockNames$Company <- NULL
StockNames$GICS_sector.ETF_type <- NULL
StockNames$GICS_industry.ETF_subtype <- NULL

# View(StockNames[SP500==TRUE])
# View(StockNames[ETF==TRUE])
# View(StockNames[ETF==TRUE & (!is.na(M6Id))])
# View(StockNames[(!is.na(M6Id))][order(M6Id)])

saveRDS(StockNames, file.path("Data","StockNames.RDS"))






