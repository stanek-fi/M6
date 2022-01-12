library(torch)
train_split = 0.8
sample_indices =sample(nrow(iris) * train_split)

# 2. Convert our input data to matrices and labels to vectors.
x_train = as.matrix(iris[sample_indices, -5])
y_train = as.numeric(iris[sample_indices, 5])
x_test = as.matrix(iris[-sample_indices, -5])
y_test = as.numeric(iris[-sample_indices, 5])

# 3. Convert our input data and labels into tensors.
x_train = torch_tensor(x_train, dtype = torch_float())
y_train = torch_tensor(y_train, dtype = torch_long())
x_test = torch_tensor(x_test, dtype = torch_float())
y_test = torch_tensor(y_test, dtype = torch_long())


# Define the network
model = nn_sequential(
  
  # Layer 1
  nn_linear(4, 8),
  nn_relu(), 
  
  # Layer 2
  nn_linear(8, 16),
  nn_relu(),
  
  # Layer 3
  nn_linear(16,3)
)

# Define cost and optimizer
criterion = nn_cross_entropy_loss()  
optimizer = optim_adam(model$parameters, lr = 0.01)

epochs = 200

# Train the net
for(i in 1:epochs){
  
  optimizer$zero_grad()
  
  y_pred = model(x_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  
  # Check Training
  if(i %% 10 == 0){
    
    winners = y_pred$argmax(dim=2)+1
    corrects = (winners == y_train)
    accuracy = corrects$sum()$item() / y_train$size()
    
    cat(" Epoch:", i,"Loss: ", loss$item()," Accuracy:",accuracy,"\n")
  }
}