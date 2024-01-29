library(ggplot2)
library(tidyverse)
library(cowplot)
library(reshape2)
library(viridis)
load('20231216_200_Metrics.RData')

#### Accuracy / Validation Accuracy, Loss / Validation Loss ####
data <- read.csv("metrics_export_df.csv")

epoch_stats <- data %>%
  group_by(epoch) %>%
  summarize(mean_accuracy = mean(accuracy),
            sd_accuracy = sd(accuracy),
            mean_val_accuracy = mean(val_accuracy),
            sd_val_accuracy = sd(val_accuracy))

epoch_stats_loss <- data %>%
  group_by(epoch) %>%
  summarize(mean_loss = mean(loss),
            sd_loss = sd(loss),
            mean_val_loss = mean(val_loss),
            sd_val_loss = sd(val_loss))


p1 <- ggplot(epoch_stats, aes(x = epoch)) +
  geom_ribbon(aes(ymin = mean_accuracy - sd_accuracy, ymax = mean_accuracy + sd_accuracy, fill = "Accuracy"), alpha = 0.2) +
  geom_ribbon(aes(ymin = mean_val_accuracy - sd_val_accuracy, ymax = mean_val_accuracy + sd_val_accuracy, fill = "Validation\nAccuracy"), alpha = 0.2) +
  geom_line(aes(y = mean_accuracy), color = "blue") +
  geom_line(aes(y = mean_val_accuracy), color = "black") +
  scale_fill_manual(values = c("Accuracy" = "blue", "Validation\nAccuracy" = "red")) +
  labs(title = "B)",
       x = "Epoch",
       y = "Accuracy",
       fill = NULL) +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.3))
p1

p2 <- ggplot(epoch_stats_loss, aes(x = epoch)) +
  geom_ribbon(aes(ymin = mean_loss - sd_loss, ymax = mean_loss + sd_loss, fill = "Loss"), alpha = 0.2) +
  geom_ribbon(aes(ymin = mean_val_loss - sd_val_loss, ymax = mean_val_loss + sd_val_loss, fill = "Validation\nLoss"), alpha = 0.2) +
  geom_line(aes(y = mean_loss), color = "green") +
  geom_line(aes(y = mean_val_loss), color = "black") +
  scale_fill_manual(values = c("Loss" = "green", "Validation\nLoss" = "orange")) +
  labs(title = "D)",
       x = "Epoch",
       y = "Loss",
       fill = NULL) +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.3))
p2
plot_grid(p1, p2)
jpeg("Acc_200_Epochs_V3.jpeg",width = 8,height =4,units = "in", res=600)
plot_grid(p1, p2)
dev.off()

#### Overall Accuracy / Overall Loss ####
seeds <- read.csv("eval_export_df.csv")
seeds$acc_loss <- ifelse(grepl("loss", seeds$X), "Loss", "Accuracy")
seeds$acc_loss <- as.factor(seeds$acc_loss)

seed_acc <- seeds[seeds$acc_loss=="Accuracy",]
seed_loss <- seeds[seeds$acc_loss=="Loss",]

seeds_avg <- seeds %>% group_by(acc_loss) %>% summarise(Value = mean(evaluation), SD = sd(evaluation))
acc_avg <- seeds_avg[1,]

loss_avg <- seeds_avg[2,]

p_acc <- seed_acc %>% ggplot(aes(x = acc_loss, y = evaluation)) +
  geom_boxplot()

dataa <- data.frame(Accuracy = 0.7144529, StdDev = 0.03009728)
lower_bound <- round(dataa$Accuracy - dataa$StdDev, 3)
upper_bound <- round(dataa$Accuracy + dataa$StdDev, 3)

p_acc <- ggplot(dataa, aes(x = "", y = Accuracy, fill = "Accuracy")) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_errorbar(aes(ymin = Accuracy - StdDev, ymax = Accuracy + StdDev), width = 0.2, position = position_dodge(0.5)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  labs(y = "Accuracy", x = NULL) +
  theme_minimal() +
  theme(legend.position = "none") +
  geom_text(aes(label = paste(round(Accuracy, 3), "\u00B1", round(StdDev, 5)), angle = 90, hjust = 2), vjust = 0.5, size = 3) +
  scale_fill_manual(values = c("red")) +
  ggtitle("A)")


p_acc

dataaa <- data.frame(Loss = 1.506443, StdDev = 0.3959727)
p_loss <- dataaa %>% ggplot(aes(x = "", y = Loss, fill = "Loss")) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_errorbar(aes(ymin = Loss - StdDev, ymax = Loss + StdDev), width = 0.2, position = position_dodge(0.5)) +
  scale_y_continuous(limits = c(0, 2), breaks = seq(0, 2, by = 0.1)) +
  labs(y = "Loss", x = NULL) +
  theme_minimal() +
  theme(legend.position = "none") +
  geom_text(aes(label = paste(round(Loss, 3), "\u00B1", round(StdDev, 3)), angle = 90, hjust = 2.3), vjust = 0.5, size = 3) +
  scale_fill_manual(values = c("orange")) +
  ggtitle("C)")


p_loss


#### Merge plots ####
plot_grid(p_acc, p1, p_loss, p2, rel_widths = c(0.7, 2, 0.7, 2), ncol = 4)
jpeg("Acc_200_Epochs_V6.jpeg",width = 10,height =4,units = "in", res=600)
plot_grid(p_acc, p1, p_loss, p2, rel_widths = c(0.7, 2, 0.7, 2), ncol = 4)
dev.off()


#### Average confusion matrix after 200 iterations of CNN ####
# Step 1: Calculate the average confusion matrix
average_conf_matrix <- Reduce("+", confused_export) / length(confused_export)

# Function to normalize the confusion matrix #
normalize_conf_matrix <- function(conf_matrix) {
  row_sums <- rowSums(conf_matrix)
  conf_matrix_normalized <- sweep(conf_matrix, 1, row_sums, FUN = "/")
  return(conf_matrix_normalized)
}

conf_matrix_normalized <- normalize_conf_matrix(average_conf_matrix)
conf_matrix_normalized <- conf_matrix_normalized * 100


# Step 2: Convert the average matrix to a long format
average_conf_matrix_long <- melt(average_conf_matrix)
conf_matrix_normalized_long <- melt(conf_matrix_normalized)

# Step 3: Create the heatmap
p3_norm <- ggplot(conf_matrix_normalized_long, aes(Actual, Predicted, fill = value)) +
  geom_tile(color = "black", size = 0.3) +  # Add black outline
  scale_fill_gradient(low = "white", high = "red") +
  labs(x = "Predicted class", y = "Actual class", fill = "Normalized\npercentage") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 2.5, vjust = 1) +
  theme_minimal() +
  coord_fixed() +
  scale_x_continuous(breaks = c(0,1,2,3,4,5)) +
  scale_y_continuous(breaks = c(0,1,2,3,4,5)) +
  ggtitle('E)')

p3_norm

#### Accuracy, precision, recall (sensitivity), F1 score for each class ####
n <- nrow(average_conf_matrix)
accuracy <- sum(diag(average_conf_matrix)) / sum(average_conf_matrix)

precision <- recall <- f1_score <- numeric(n)

for (i in 1:n) {
  TP <- average_conf_matrix[i, i]
  FP <- sum(average_conf_matrix[-i, i])
  FN <- sum(average_conf_matrix[i, -i])
  TN <- sum(average_conf_matrix[-i, -i])
  
  precision[i] <- TP / (TP + FP)
  recall[i] <- TP / (TP + FN)
  f1_score[i] <- ifelse((precision[i] + recall[i]) == 0, 0,
                        2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
}

# Put metrics into data frame #
metrics <- c(rep('Precision', 6), rep('Recall', 6), rep('F1 Score', 6))
category <- c(rep(c(0:5), 3))
values <- c(precision, recall, f1_score)
metrics_df <- data.frame(values, category, metrics)
metrics_df$category <- as.factor(metrics_df$category)
metrics_df$metrics <- factor(metrics_df$metrics, levels = c('Precision', 'Recall', 'F1 Score'))

p4 <- ggplot(metrics_df, aes(x=category, y=values, fill=category)) +
  geom_bar(stat="identity", position=position_dodge()) +
  scale_x_discrete("Senescence category") +
  scale_y_continuous(NULL, breaks = c(0,0.2,0.4,0.6,0.8,1.0,1.2)) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none",
        legend.justification = c("right", "top"),
        axis.text.x = element_text(angle = 0, vjust = 1, hjust = 0.5)) +
  geom_text(aes(label=round(values, 3)),
            position=position_dodge(width=.9),
            vjust=0.5,
            hjust=1.1,
            angle=90,
            color="white",
            size=4) +
  scale_fill_viridis_d() +
  facet_wrap(~metrics) +
  ggtitle('F)')

p4


#### Merge all plots - Normalized confusion matrix ####
top <- plot_grid(p_acc, p1, p_loss, p2, rel_widths = c(0.7, 2, 0.7, 2), ncol = 4)
bottom <- plot_grid(p3_norm, p4, rel_widths = c(1.3, 2), ncol = 2)

jpeg("Acc_200_Epochs_V12.jpeg",width = 11, height =8, units = "in", res=600)
plot_grid(top, bottom, nrow = 2)
dev.off()