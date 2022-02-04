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