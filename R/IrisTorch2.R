library(torch)
train_split = 0.8
# sample_indices = sample(nrow(iris) * train_split)
sample_indices = sample(1:nrow(iris), size=nrow(iris) * train_split)
# sample_indices =1:nrow(iris) %in% sample_indices
# sample_indices =sample(c(T,F),prob=c(0.8,0.2),size=nrow(iris),replace = T)

# 2. Convert our input data to matrices and labels to vectors.
irisMod <- iris

type = as.factor(sample(1:2,size=nrow(irisMod),replace = T))
xtype <- model.matrix(~type-1)

x_train = as.matrix(irisMod[sample_indices, -5])
y_train = as.numeric(irisMod[sample_indices, 5])
x_test = as.matrix(irisMod[-sample_indices, -5])
y_test = as.numeric(irisMod[-sample_indices, 5])
xtype_train = as.matrix(xtype[sample_indices,])
xtype_test = as.matrix(xtype[-sample_indices,])

# 3. Convert our input data and labels into tensors.
x_train = torch_tensor(x_train, dtype = torch_float())
y_train = torch_tensor(y_train, dtype = torch_long())
x_test = torch_tensor(x_test, dtype = torch_float())
y_test = torch_tensor(y_test, dtype = torch_long())
xtype_train = torch_tensor(xtype_train, dtype = torch_float())
xtype_test = torch_tensor(xtype_test, dtype = torch_float())




# net = nn_module(
#   initialize = function() {
#     self$lin1 <- nn_linear(4, 8)
#     self$lin2 <- nn_linear(8, 16)
#     self$lin3 <-nn_linear(16,3)
#     self$afun <- nn_relu()
#   },
#   forward = function(x) {
#     x <- self$afun(self$lin1(x))
#     x <- self$afun(self$lin2(x))
#     x <- self$lin3(x)
#     x
#   }
# )

net = nn_module(
  initialize = function() {
    self$lin1 <- nn_linear(4, 3, bias = F)
  },
  forward = function(x,xtype) {
    x <- self$lin1(x)
    x
  }
)


net = nn_module(
  initialize = function() {
    self$lin1 <- nn_linear(4, 3, bias = F)
    self$linIn <- nn_linear(2, 12, bias = F)
  },
  forward = function(x,xtype) {
    theta <- self$linIn(xtype)
    theta <- theta[1,]
    theta <- torch_reshape(theta,c(3,4))

    state_dict <- self$lin1$state_dict()
    state_dict[[1]] <- theta
    # state_dict[[1]] <- state_dict[[1]] * 0.99
    self$lin1$load_state_dict(state_dict)

    x <- self$lin1(x)
    x
  }
)




lin1 <- nn_linear(4, 3, bias = F)
linIn <- nn_linear(2, 12)
theta <- linIn(xtype_train[1:4,])
theta <- theta[1,] 
theta <- torch_reshape(theta,c(3,4))


# lin1$parameters[[1]] <- theta

state_dict = lin1$state_dict()
transformed_param <- state_dict[[1]] *  0.9
transformed_param <- theta 
state_dict[[1]]=transformed_param 
lin1$load_state_dict(state_dict)

lin1(x_train)






# Define cost and optimizer
model <- net()

criterion = nn_cross_entropy_loss()  
optimizer = optim_adam(model$parameters, lr = 0.01)

epochs = 30

# Train the net
for(i in 1:epochs){
  
  model$parameters
  
  optimizer$zero_grad()
  
  y_pred = model(x_train,xtype_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  
  # Check Training
  # if(i %% 10 == 0){
  if(T){
    
    winners = y_pred$argmax(dim=2)+1
    corrects = (winners == y_train)
    accuracy = corrects$sum()$item() / y_train$size()
    
    cat(" Epoch:", i,"Loss: ", loss$item()," Accuracy:",accuracy,"\n")
  }
}

model$parameters