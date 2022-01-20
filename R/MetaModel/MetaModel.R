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


MetaModel = nn_module(
  initialize = function(baseModel, xtype, mesaParameterSize = 1) {
    self$fforward <- baseModel$fforward
    self$stateStructure <- baseModel$stateStructure
    self$flattenState <- baseModel$flattenState
    self$unflattenState <- baseModel$unflattenState
    self$stateSize <- baseModel$stateSize
    self$xtypeSize <- ncol(xtype)
    self$mesaParameterSize <- mesaParameterSize
    
    self$mesaLayerWeight <- nn_parameter(torch_tensor(matrix(0, nrow = self$mesaParameterSize, ncol = self$xtypeSize), dtype = torch_float()))
    self$metaLayer <- nn_linear(mesaParameterSize, baseModel$stateSize, bias = F)
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
        self$mesaParameterSize <- metaModel$mesaParameterSize
        self$metaLayerBias <- torch_tensor(as.array(metaState$metaLayerBias),dtype = torch_float()) #nasty trick to make it leaf
        self$metaLayerWeight <- torch_tensor(as.array(metaState$metaLayer.weight),dtype = torch_float()) #nasty trick to make it leaf
        
        self$mesaState <- nn_parameter(torch_tensor(rep(0, self$mesaParameterSize), dtype = torch_float()))
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