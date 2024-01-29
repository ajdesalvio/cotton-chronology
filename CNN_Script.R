remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
install_tensorflow(envname = "r-tensorflow")

library(tidyverse)
library(dplyr)
library(readxl)
library(EBImage)
library(keras)
library(tensorflow)


# Choose file import type
pic_type <- ".jpg"; div <- 1
pic_type <- ".tif"; div <- 255 # All of the display functions will account for tif or jpeg now

# Import scores
scores <- read_xlsx("Senescence.xlsx", sheet = 1)
scores_clean <- na.omit(scores, cols = "Score")
categ_num <- 6 # Number of classification categories

# Explore
pics <- list.files(pattern = paste0("\\", pic_type, "$")) # This will list all JPEG or TIF files, change WD to TIFs folder
mypic <- lapply(pics, readImage) # Import all images efficiently with lapply()
print(mypic[1])
display(mypic[[1]] / div) # Divide by 255 if viewing a TIFF, the pic_type toggle above will accomplish this
summary(mypic[[1]])
hist(mypic[[1]])
str(mypic[[1]]); str(mypic[[2]]); str(mypic[[3]]) # Average image dimensions seem to be 163 x 163 for JPEGs
pic_dim <- 163 # Set picture dimensions to be used throughout script

# Resize
mypic2 <- mypic
names(mypic2) <- pics

mypic2 <- lapply(mypic2, function(x) resize(x, pic_dim, pic_dim))

display(mypic2[[1]] / div)
str(mypic2[[1]]) # Now the image dimensions are all pic_dim x pic_dim

# Reshape
mypic2 <- lapply(mypic2, function(x) array_reshape(x, c(pic_dim, pic_dim, 3))) # x is each element of the list
display(mypic2[[1]] / div) # Note that the three layers have been split up

#### OGR ####
OGR_files <- scores_clean$File_Name # Use this for JPEGs

OGR_files <- scores_clean$File_Name_TIFs # Use this for TIFs

mypic_OGR <- mypic2[OGR_files]; setequal(names(mypic_OGR), OGR_files)
totalsize <- length(unique(scores_clean$Pltg_ID_2)); totalsize # Number of unique pedigrees left after filtering NAs
trainsize <- totalsize*0.8; trainsize # 188 pedigrees training, 47 test

OGR_train_df <- as.data.frame(unique(scores_clean$Pltg_ID_2))
colnames(OGR_train_df) <- "Pltg_ID_2" # Used later for randomly partitioning by genotype


# Function to convert a list of arrays into a 4D array
convert_to_4d_array <- function(list_of_arrays) {
  # Number of images
  num_images <- length(list_of_arrays)
  
  # Initialize an array to hold the stacked images
  array_4d <- array(dim = c(num_images, pic_dim, pic_dim, 3))
  
  # Stack each 3D array (image) into the 4D array
  for (i in 1:num_images) {
    array_4d[i,,,] <- list_of_arrays[[i]]
  }
  
  return(array_4d)
}


# Change seed to run different train/test splits
eval_list <- list()
metrics_list <- list()
confused_list <- list()

num_reps <- 1

for (i in 1:num_reps) {
  
  set.seed(as.numeric(i))

  OGR_train <- sample(OGR_train_df$Pltg_ID_2, trainsize, replace = F) # 188 unique training pedigrees
  OGR_test <- OGR_train_df %>% filter(!Pltg_ID_2 %in% OGR_train) # 47 unique testing pedigrees
  OGR_test <- as.vector(OGR_test$Pltg_ID_2)

  OGR_train_jpg <- paste0(OGR_train, pic_type)
  OGR_test_jpg <- paste0(OGR_test, pic_type)

  OGR_train_indices <- unique(unlist(sapply(OGR_train_jpg, function(v) grep(v, names(mypic_OGR), value = TRUE)))) %>% as.vector()
  OGR_test_indices <- unique(unlist(sapply(OGR_test_jpg, function(v) grep(v, names(mypic_OGR), value = TRUE)))) %>% as.vector()


#### Run this for TIFs ####
  trainx <- mypic_OGR[OGR_train_indices]
  trainy_unordered <- scores_clean %>% filter(File_Name_TIFs %in% OGR_train_indices)
  trainy <- trainy_unordered %>% 
    mutate(order = match(File_Name_TIFs, OGR_train_indices)) %>% 
    arrange(order) %>% 
    select(-order)
  trainy <- trainy$Score
  trainy_table <- table(trainy)

  testx <- mypic_OGR[OGR_test_indices]
  testy_unordered <- scores_clean %>% filter(File_Name_TIFs %in% OGR_test_indices)
  testy <- testy_unordered %>% 
    mutate(order = match(File_Name_TIFs, OGR_test_indices)) %>% 
    arrange(order) %>% 
    select(-order)
  testy <- testy$Score


  # JPEGs: Convert trainx and testx to 4D arrays
  trainx_4d <- convert_to_4d_array(trainx)
  testx_4d <- convert_to_4d_array(testx)

  # Required step if using TIFS: Normalize the pixel values to the range [0, 1]
  trainx_4d <- trainx_4d / 255
  testx_4d <- testx_4d / 255

  # One-hot encode trainy and testy
  trainy <- to_categorical(trainy, num_classes = categ_num)
  testy <- to_categorical(testy, num_classes = categ_num)


  #### Basic CNN without class weights ####
  model <- keras_model_sequential() %>%
    # Convolutional layers
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(pic_dim, pic_dim, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
    # Flatten the output to feed into dense layers
    layer_flatten() %>%
  
    # Dense layers
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = categ_num, activation = 'softmax')

  # Compile the model
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(lr = 0.0001),
    metrics = 'accuracy')

  # Fit the model
  history <- model %>%
    fit(
      trainx_4d,
      trainy,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2)


  #### Model evaluation ####
  # Evaluate the model
  evaluation <- model %>% evaluate(testx_4d, testy)
  evaluation_df <- as.data.frame(evaluation)
  evaluation_df$seed <- as.numeric(i)

  # Predictions
  predictions <- model %>% predict(testx_4d)
  predicted_classes <- k_argmax(predictions)
  predicted_classes <- as.vector(predicted_classes)

  # Convert testy_hot back to actual classes
  # The maximum index in each row gives the class label
  actual_classes <- apply(testy, 1, which.max)

  # Create a confusion matrix
  confusion_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)
  rownames(confusion_matrix) <- c(0:5)

  # Convert the history metrics to a data frame
  history_df <- as.data.frame(history$metrics)

  # Add an 'epoch' column
  history_df$epoch <- seq_len(nrow(history_df))

  # Add a column for the seed
  history_df$seed <- as.numeric(i)
  
  
  #### Exports ####
  eval_list[[i]] <- evaluation_df
  metrics_list[[i]] <- history_df
  confused_list[[i]] <- confusion_matrix

}

eval_export_df <- do.call(rbind, eval_list)
metrics_export_df <- do.call(rbind, metrics_list)