# https://willhipson.netlify.app/post/torch-for-r/torch_stars/

library(tidyverse)
library(torch)

# library(pins)
# pin_name <- pin_find("star-type-classification", board = "local")[[1]]
# pin_dir <- pin_get(pin_name, board = "kaggle")

stars_raw <- read_csv("Data/Stars.csv")

head(stars_raw)


fix_color <- function(x, n = 5) {
  x_lower <- str_to_lower(x)
  x_fix <- str_replace(x_lower, " |-", "_") # replaces either blank space or '-' with '_'
  x_lump <- fct_lump(x_fix, n, other_level = "other")
  x_lump
}

# Test
stars_raw %>%
  mutate(Color = fix_color(Color)) %>%
  count(Color, sort = TRUE)



torch_rand(2, 3)
mat <- matrix(c(1, 4, 7,
                2, 5, 2), nrow = 2, byrow = TRUE)
my_tensor <- torch_tensor(mat)
my_tensor



stars_dataset <- dataset(
  
  name = "stars_dataset",
  
  initialize = function(df) {
    
    # Data preprocessing
    stars_pp <- df %>%
      select(-Type) %>%
      mutate(across(c("Temperature", "L", "R"), log10),
             Color = fix_color(Color))
    
    # Numeric predictors
    self$x_num <- stars_pp %>%
      select(where(is.numeric)) %>%
      as.matrix() %>%
      torch_tensor()
    
    # Categorical predictors
    self$x_cat <- stars_pp %>%
      select(!where(is.numeric)) %>%
      mutate(across(everything(), ~as.integer(as.factor(.x)))) %>%
      as.matrix() %>%
      torch_tensor()
    
    # Target data
    type <- as.integer(df$Type) + 1
    self$y <- torch_tensor(type)
  },
  
  .getitem = function(i) {
    x_num <- self$x_num[i, ]
    x_cat <- self$x_cat[i, ]
    y <- self$y[i]
    
    list(x = list(x_num, x_cat),
         y = y)
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
)




set.seed(941843)

length_ds <- length(stars_dataset(stars_raw))

train_id <- sample(1:length_ds, ceiling(0.80 * length_ds))
valid_id <- setdiff(1:length_ds, train_id)

# Datasets
train_ds <- stars_dataset(stars_raw[train_id, ])
valid_ds <- stars_dataset(stars_raw[valid_id, ])


# train_ds[4]


# Dataloaders
train_dl <- train_ds %>%
  dataloader(batch_size = 25, shuffle = TRUE)

valid_dl <- valid_ds %>%
  dataloader(batch_size = 25, shuffle = FALSE)




embedding_mod <- nn_module(
  
  initialize = function(levels) {
    
    self$embedding_modules = nn_module_list(
      map(levels, ~nn_embedding(.x, embedding_dim = ceiling(.x/2)))
    )
  },
  
  forward = function(x) {
    
    embedded <- vector("list", length(self$embedding_modules))
    for(i in 1:length(self$embedding_modules)) {
      # gets the i-th embedding module and calls the function on the i-th column
      # of the tensor x
      embedded[[i]] <- self$embedding_modules[[i]](x[, i])
    }
    
    torch_cat(embedded, dim = 2)
  }
)


net <- nn_module(
  
  "stars_net",
  
  initialize = function(levels, n_num_col) {
    
    self$embedder <- embedding_mod(levels)
    
    # calculate dimensionality of first fully-connected layer
    embedding_dims <- sum(ceiling(levels / 2))
    
    self$fc1 <- nn_linear(in_features = embedding_dims + n_num_col,
                          out_features = 32)
    self$fc2 <- nn_linear(in_features = 32,
                          out_features = 16)
    self$output <- nn_linear(in_features = 16,
                             out_features = 6) # number of Types
  },
  
  forward = function(x_num, x_cat) {
    
    embedded <- self$embedder(x_cat)
    
    predictors <- torch_cat(list(x_num$to(dtype = torch_float()), embedded), dim = 2)
    
    predictors %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$output() %>%
      nnf_softmax(2) # along 2nd dimension (i.e., rowwise)
  }
)


levels <- stars_raw %>%
  mutate(Color = fix_color(Color)) %>%
  select(!where(is.numeric)) %>%
  map_dbl(n_distinct) %>%
  unname()

n_num_col <- ncol(train_ds$x_num)



device <- if(cuda_is_available()) {
  torch_device("cuda:0")
} else {
  "cpu"
}

model <- net(levels, n_num_col)

model <- model$to(device = device)






torch_manual_seed(414412)
optimizer <- optim_asgd(model$parameters, lr = 0.025)
n_epochs <- 140

for(epoch in 1:n_epochs) {
  
  # set the model to train
  model$train()
  
  train_losses <- c()
  
  # Make prediction, get loss, backpropagate, update weights
  coro::loop(for (b in train_dl) {
    optimizer$zero_grad()
    output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
    loss <- nnf_cross_entropy(output, b$y$to(dtype = torch_long(), device = device))
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
  })
  
  # Evaluate
  model$eval()
  
  valid_losses <- c()
  valid_accuracies <- c()
  
  coro::loop(for (b in valid_dl) {
    output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
    loss <- nnf_cross_entropy(output, b$y$to(dtype = torch_long(), device = device))
    valid_losses <- c(valid_losses, loss$item())
    pred <- torch_max(output, dim = 2)[[2]]
    correct <- (pred == b$y)$sum()$item()
    valid_accuracies <- c(valid_accuracies, correct/length(b$y))
  })
  
  if(epoch %% 10 == 0) {
    cat(sprintf("Epoch %d: train loss: %3f, valid loss: %3f, valid accuracy: %3f\n",
                epoch, mean(train_losses), mean(valid_losses), mean(valid_accuracies)))
  }
}



library(yardstick)

pred <- torch_max(model(valid_ds$x_num, valid_ds$x_cat), dim = 2)[[2]] %>%
  as_array()

truth <- valid_ds$y %>%
  as_array()

type_levels <- c("Red Dwarf", "Brown Dwarf", "White Dwarf",
                 "Main Sequence", "Super Giant", "Hyper Giant")

confusion <- bind_cols(pred = pred, truth = truth) %>%
  mutate(across(everything(), ~factor(.x, levels = 1:6, labels = type_levels))) %>%
  conf_mat(truth, pred)

autoplot(confusion, type = "heatmap") + 
  scale_fill_distiller(palette = 2, direction = "reverse")










