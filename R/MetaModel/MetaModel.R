prepareBaseModel <- function(baseModel, x) {
  state <- baseModel$state_dict()
  
  baseModel$eval()
  output <- baseModel(x)
  outputControl <- baseModel$fforward(x,state)
  meanDiff <- as.array(mean(abs(output - outputControl)))
  if(meanDiff>1e-10){
    stop("Supplied fforward method of the baseModel does not match the forward method.")
  }
  baseModel$train()
  
  baseModel$outputSize <- output$size(2)
  
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
  initialize = function(baseModel, xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0) {
    self$fforward <- baseModel$fforward
    self$stateStructure <- baseModel$stateStructure
    self$flattenState <- baseModel$flattenState
    self$unflattenState <- baseModel$unflattenState
    self$outputSize <- baseModel$outputSize 
    self$stateSize <- baseModel$stateSize
    self$xtypeSize <- ncol(xtype)
    self$mesaParameterSize <- mesaParameterSize
    self$allowBias <- allowBias
    self$mesaLayerWeight <- nn_parameter(torch_tensor(matrix(0, nrow = self$mesaParameterSize, ncol = self$xtypeSize), dtype = torch_float()))
    self$metaLayer <- nn_linear(mesaParameterSize, baseModel$stateSize, bias = F)
    if(self$allowBias){
      self$metaLayerBias <- nn_parameter(torch_tensor(rep(0, self$stateSize), dtype = torch_float()))
    }
    self$baseState <- as.array(self$flattenState(baseModel$state_dict())) #storing it array to avoid odd ourside reference error. Update...it is caused by inbaility to conect references after saving the model...probably not worth fixing
    self$dropout <- nn_dropout(p = pDropout)
  },
  
  forward = function(x,xtype) {
    xout <- torch_zeros(nrow(x),self$outputSize)
    iscoalesced = try({xtype$is_coalesced()},silent = T)
    baseState <- torch_tensor(self$baseState)
    
    if(iscoalesced == TRUE){
      columns <- xtype$indices()[2,]
      uniqueColumns <- unique(as.array(columns))
      for(i in uniqueColumns){
        indices <- columns == i
        xtypei <- torch_tensor(replace(numeric(xtype$size(2)), i + 1, 1)) #must add one to correct for zero indexing
        mesaState <- nnf_linear(xtypei, self$mesaLayerWeight)
        if(self$allowBias){
          metaStateFlatDiff <- self$metaLayerBias + self$metaLayer(mesaState)
        }else{
          metaStateFlatDiff <- self$metaLayer(mesaState)
        }
        metaStateFlat <-  baseState + self$dropout(metaStateFlatDiff)
        metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
        xout[indices] <- self$fforward(x[indices,],metaState)
      }
    }else{
      # for(i in seq_len(ncol(xtype))){
      #   indices <- xtype[,i]>0
      #   if(as.numeric(indices$max())>0){
      #     mesaState <- nnf_linear(xtype[indices,][1,], self$mesaLayerWeight)
      #     metaStateFlat <-  self$metaLayerBias + self$metaLayer(mesaState)
      #     metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
      #     xout[indices] <- self$fforward(x[indices,],metaState)
      #   }
      # }
      stop("Currently works only with sparse xtype.")
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
        self$allowBias <- metaModel$allowBias
        if(self$allowBias){
          self$metaLayerBias <- torch_tensor(as.array(metaState$metaLayerBias),dtype = torch_float()) #nasty trick to make it leaf
        }
        self$metaLayerWeight <- torch_tensor(as.array(metaState$metaLayer.weight),dtype = torch_float()) #nasty trick to make it leaf
        self$baseState <- metaModel$baseState
        self$mesaState <- nn_parameter(torch_tensor(rep(0, self$mesaParameterSize), dtype = torch_float()))
      },
      forward = function(x) {
        mesaState <- self$mesaState
        baseState <- torch_tensor(self$baseState)
        if(self$allowBias){
          metaStateFlatDiff <- self$metaLayerBias + nnf_linear(mesaState,self$metaLayerWeight)
        }else{
          metaStateFlatDiff <- nnf_linear(mesaState,self$metaLayerWeight)
        }
        metaStateFlat <-  baseState + metaStateFlatDiff
        metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
        self$fforward(x,metaState)
      }
    )
  }
)