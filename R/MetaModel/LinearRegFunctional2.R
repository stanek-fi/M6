library(torch)
rm(list=ls())

N <- 1000
NumGroups <- 2
# thetas <- rep(1,NumGroups)
thetas <- c(1,2)
train_split = 0.8

type = as.factor(sample(1:NumGroups,size=N ,replace = T))
xtype <- model.matrix(~type-1)
thetas <- xtype %*% thetas
x <- as.matrix(rnorm(N),N,1)
y <- x * thetas

sample_indices = sample(1:N, size=N * train_split)
x_train = as.matrix(x[sample_indices,])
y_train = as.matrix(y[sample_indices,])
x_test = as.matrix(x[-sample_indices,])
y_test = as.matrix(y[-sample_indices,])
xtype_train = as.matrix(xtype[sample_indices,])
xtype_test = as.matrix(xtype[-sample_indices,])


x_train = torch_tensor(x_train, dtype = torch_float())
y_train = torch_tensor(y_train, dtype = torch_float())
x_test = torch_tensor(x_test, dtype = torch_float())
y_test = torch_tensor(y_test, dtype = torch_float())
xtype_train = torch_tensor(xtype_train, dtype = torch_float())
xtype_test = torch_tensor(xtype_test, dtype = torch_float())

BaseModel = nn_module(
  initialize = function() {
    self$lin1 <- nn_linear(1, 1)
  },
  forward = function(x) {
    x <- self$lin1(x)
    x
  },
  fforward = function(x,state) {
    x <- nnf_linear(x,state$lin1.weight,state$lin1.bias)
    x
  }
)

prepareBaseModel <- function(baseModel, x = NULL) {
  state <- baseModel$state_dict()
  
  if(!is.null(x)){
    meanDiff <- as.array(mean(abs(baseModel(x) - baseModel$fforward(x,state))))
    if(meanDiff>1e-10){
      stop("Supplied fforward method of the baseModel does not match the forward method.")
    }
  }
  
  baseModel$stateStructure <- rapply(state, function(x) dim(x), how = "list")
  
  baseModel$flattenState <- function(state) {
    torch_cat(rapply(state, function(x) x$view(-1), how = "unlist"))
  }
  
  baseModel$unflattenState <- function(stateFlat, stateStructure) {
    counter <- 1
    rapply(stateStructure , function(dimensions){
      totsize <- prod(dimensions)
      out <- stateFlat[(counter):(counter - 1 + totsize)]$view(dimensions)
      counter <<- counter + totsize
      return(out)
    }, how = "list")
  }
  
  stateControl <- baseModel$unflattenState(baseModel$flattenState(state), baseModel$stateStructure)
  stateControlEqual <- unlist(mapply(function(x,y) {
    all.equal(as.array(x), as.array(y))
  }, x=state, y=stateControl), recursive = T)
  if(!all(stateControlEqual == "TRUE")){
    stop("State transformation functions failed.")
  }
  
  baseModel$stateSize <- baseModel$flattenState(state)$size()

  return(baseModel)
}

baseModel <- BaseModel()
baseModel <- prepareBaseModel(baseModel)

criterion = nn_mse_loss()
optimizer = optim_adam(baseModel$parameters, lr = 0.01)
epochs = 300
for(i in 1:epochs){
  optimizer$zero_grad()
  y_pred = baseModel(x_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  # Check Training
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}



# temp <- nn_linear(100, 1000)
# hist(as.array(temp$state_dict()$weight))
# temp(xtype_test)

x <- x_train
xtype <- xtype_train
self <- list()
mesaParameters = 3

MetaModel = nn_module(
  initialize = function(baseModel, xtype, mesaParameters = 1) {
    self$fforward <- baseModel$fforward
    self$stateStructure <- baseModel$stateStructure
    self$flattenState <- baseModel$flattenState
    self$unflattenState <- baseModel$unflattenState
    self$stateSize <- baseModel$stateSize
    self$xtypeSize <- ncol(xtype)
    self$mesaParameters <- mesaParameters
    
    self$mesaLayerWeight <- nn_parameter(torch_tensor(matrix(0, nrow = self$mesaParameters, ncol = self$xtypeSize), dtype = torch_float()))
    self$metaLayer <- nn_linear(mesaParameters, baseModel$stateSize, bias = F)
    self$metaLayerBias <- nn_parameter(torch_tensor(as.array(self$flattenState(baseModel$state_dict())), dtype = torch_float())) #nasty trick to make it leaf
  },
  
  forward = function(x,xtype) {
      xout <- torch_zeros(nrow(x),1)
      for(i in seq_len(ncol(xtype))){
        indices <- xtype[,i]>0
        if(as.numeric(indices$max())>0){
          mesaState <- nnf_linear(xtype[indices,][1,], self$mesaLayerWeight)
          metaStateFlat <-  self$metaLayerBias + self$metaLayer(mesaState)
          metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
          xout[indices] <- self$fforward(x[indices,],metaState)
        }
      }
      xout
  },
  
  MesaModel = function(metaModel) {
    
    metaState <- metaModel$state_dict()
    
    nn_module(
      initialize = function() {
        self$fforward <- metaModel$fforward
        self$stateStructure <- metaModel$stateStructure
        self$unflattenState <- metaModel$unflattenState
        self$flattenState <- metaModel$flattenState
        self$mesaParameters <- metaModel$mesaParameters
        # self$metaLayerBias <- metaState$metaLayerBias
        self$metaLayerBias <- torch_tensor(as.array(metaState$metaLayerBias),dtype = torch_float()) #nasty trick to make it leaf
        # self$metaLayerWeight <- metaState$metaLayer.weight
        self$metaLayerWeight <- torch_tensor(as.array(metaState$metaLayer.weight),dtype = torch_float()) #nasty trick to make it leaf
        
        self$mesaState <- nn_parameter(torch_tensor(rep(0, self$mesaParameters), dtype = torch_float()))
      },
      forward = function(x) {
        mesaState <- self$mesaState
        metaStateFlat <-  self$metaLayerBias + nnf_linear(mesaState,self$metaLayerWeight)
        metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
        self$fforward(x,metaState)
      }
    )
  }
)

metaModel <- MetaModel(baseModel, xtype, mesaParameters = 3)
criterion = nn_mse_loss()
optimizer = optim_adam(metaModel$parameters, lr = 0.01)
epochs = 300

for(i in 1:epochs){
  
  optimizer$zero_grad()
  y_pred = metaModel(x_train,xtype_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  # Check Training
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}

mesaModel <- metaModel$MesaModel(metaModel)()
criterion = nn_mse_loss()
optimizer = optim_adam(mesaModel$parameters, lr = 0.01)
epochs = 300

rowsubset <- xtype_train[,2]==1
x_train_subset <- x_train[rowsubset,]
y_train_subset <- y_train[rowsubset]
for(i in 1:epochs){
  
  optimizer$zero_grad()
  y_pred = mesaModel(x_train_subset)
  loss = criterion(y_pred, y_train_subset)
  loss$backward()
  optimizer$step()
  
  # Check Training
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}

metaModel$state_dict()$mesaLayerWeight
mesaModel$state_dict()
















# 
# 
# MetaModel = nn_module(
#   initialize = function(baseModel, xtype, mesaParameters = 1, baseModelInicilization = T) {
#     self$fforward <- baseModel$fforward
#     self$stateStructure <- baseModel$stateStructure
#     self$flattenState <- baseModel$flattenState
#     self$unflattenState <- baseModel$unflattenState
#     self$stateSize <- baseModel$stateSize
#     self$xtypeSize <- ncol(xtype)
# 
#     self$mesaLayer <- nn_linear(ncol(xtype), mesaParameters, bias = FALSE)
#     self$metaLayer <- nn_linear(mesaParameters, baseModel$stateSize)
#   },
# 
#   forward = function(x,xtype) {
#     xout <- torch_zeros(nrow(x),1)
#     for(i in seq_len(ncol(xtype))){
#       indices <- xtype[,i]>0
#       if(as.numeric(indices$max())>0){
#         mesaState <- self$mesaLayer(xtype[indices,][1,])
#         metaStateFlat <- self$metaLayer(mesaState)
#         metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
#         xout[indices] <- self$fforward(x[indices,],metaState)
#       }
#     }
#     xout
#   }
# )