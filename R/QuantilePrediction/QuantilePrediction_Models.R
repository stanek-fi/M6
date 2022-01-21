FFNN <- function(x,y){
  
  x_train = torch_tensor(as.matrix(x), dtype = torch_float())
  y_train = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())
  
  Model = nn_module(
    initialize = function(numImputs) {
      self$lin1 <- nn_linear(numImputs, 8)
      self$lin2 <- nn_linear(8, 16)
      self$lin3 <- nn_linear(16, 5)
    },
    forward = function(x) {
      x <- nnf_relu(self$lin1(x))
      x <- nnf_relu(self$lin2(x))
      x <- nnf_softmax(self$lin3(x),2)
      x
    }
  )
  model <- Model(numImputs = x_train$size(2))
  
  criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}
  optimizer = optim_adam(model$parameters, lr = 0.01)
  epochs = 200
  for(i in 1:epochs){
    optimizer$zero_grad()
    
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss$backward()
    optimizer$step()
    
    if(i %% 10 == 0){
      cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
    }
  }
  
  return(model)
  
}