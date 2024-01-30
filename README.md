Welcome! In this repository, there are two scripts and a few supporting files.

`CNN_Script.R` allows you to reproduce the results reported in the manuscript. The exact parameters used to train the CNN along with all procedures to ensure accurate pairing of training and testing data are provided.

`Figures.R` allows you to reproduce the figures. In addition, the methods used to calculate accuracy, precision, recall, and F1 scores are shown. To run this script, the following files are needed: `20231216_200_Metrics.RData`, which contains the confusion matrices, accuracy/validation accuracy results, and loss/validation loss results from 200 iterations of running the CNN for 50 epochs; `metrics_export_df.csv`, which is a CSV file containing accuracy/validation accuracy results and loss/validation loss results that tracked how accuracy and loss changed with different train/test splits (50 epochs times 200 iterations); and `eval_export_df.csv`, which contains overall loss and accuracy values for each train/test split. `Senescence.xlsx` contains ground truth senescence scores for each single-plant drone image.

All of the images required to run the script are saved within `TIFs_1.zip`, `TIFs_2.zip`, and `TIFs_3.zip`. Unzip these files, move all images into a common `TIFs` folder, and then set your R working directory to this folder to ensure the CNN has access to all required training and validation images. There should be 3,290 TIFs in total.
