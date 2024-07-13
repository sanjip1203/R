# Load necessary libraries
library(tidyverse)
library(caret)
library(xgboost)
library(lightgbm)
library(e1071)
library(neuralnet)
library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape2)
library(ggcorrplot)
library(corrplot)
library(lattice)
library(rpart)
library(randomForest)
library(glmnet)


# Load the data
credit_data <- read.csv("/Users/sandipmahata/Desktop/statistical referencing and modeling/group project/credit_data.csv")

# View the first few rows of the dataset
head(credit_data)

# Display dataset information
str(credit_data)

# Summary statistics for numeric columns
summary(credit_data)

# Check for duplicates and missing values
duplicates <- sum(duplicated(credit_data))
missing_values <- colSums(is.na(credit_data))

# Print duplicates and missing values
print(paste("Duplicates: ", duplicates))
print("Missing Values:")
print(missing_values)

# Impute missing values for Occupation where Employment Profile is "Unemployed"
credit_data$Occupation[credit_data$`Employment Profile` == "Unemployed" & is.na(credit_data$Occupation)] <- "None"

# Convert 'Number of Existing Loans' to factor
credit_data$`Number.of.Existing.Loans` <- as.factor(credit_data$`Number.of.Existing.Loans`)

# Confirm there are no more missing values
missing_values <- colSums(is.na(credit_data))
print("Missing Values After Imputation:")
print(missing_values)

# Investigate the structure and shape of the dataset
print(dim(credit_data))
print(colnames(credit_data))

# Plot the distribution of Age
ggplot(credit_data, aes(x = as.factor(Age))) +
  geom_bar() +
  ggtitle("Distribution of Age") +
  xlab("Age") +
  ylab("Frequency")

# Plot the distribution of Gender
ggplot(credit_data, aes(x = Gender)) +
  geom_bar() +
  ggtitle("Distribution of Gender") +
  ylab("Frequency")

# Create the histogram plot for Income
ggplot(credit_data, aes(x = Income)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
  ggtitle("Distribution of Income") +
  xlab("Income") +
  ylab("Frequency") +
  theme_minimal()

# Create the bar plot for Number of Existing Loans
ggplot(credit_data, aes(x = `Number.of.Existing.Loans`)) +
  geom_bar(fill = "blue", color = "black") +
  labs(title = "Distribution of Number of Existing Loans",
       x = "Number of Existing Loans",
       y = "Frequency") +
  theme_minimal()

# Distribution of loan amount 
ggplot(credit_data, aes(x = Loan.Amount)) +
  geom_bar() +
  ggtitle("Distribution of Loan Amount") +
  ylab("Frequency")

# Distribution of Existing customer
ggplot(credit_data, aes(x = Existing.Customer)) +
  geom_bar() +
  ggtitle("Distribution of Existing Customer") +
  ylab("Frequency")

# Distribution of profile Score 
ggplot(credit_data, aes(x = Profile.Score)) +
  geom_bar() +
  ggtitle("Distribution of Profile Score") +
  ylab("Frequency")

# Distribution of Occupation 
ggplot(credit_data, aes(x = Occupation)) +
  geom_bar() +
  ggtitle("Distribution of Occupation") +
  ylab("Frequency")

# Box plot of each column
numeric_columns <- c("Age", "Income", "Credit.Score", "Credit.History.Length", "Number.of.Existing.Loans", "Loan.Amount", "Loan.Tenure", "LTV.Ratio", "Profile.Score")

# Create a list to store the individual plots
plots <- list()

# Generate the box plots
for (col in numeric_columns) {
  p <- ggplot(credit_data, aes_string(y = col)) +
    geom_boxplot() +
    labs(title = paste('Box Plot of', col), x = col) +
    theme_minimal()
  plots <- c(plots, list(p))
}

# Arrange the plots in a 3x3 grid
grid.arrange(grobs = plots, ncol = 3)

# Create the box plot to show distribution of Income by State
ggplot(credit_data, aes(x = State, y = Income)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  ggtitle("Distribution of Income by State") +
  xlab("State") +
  ylab("Income") +
  theme_minimal()

# Compute the correlation matrix
cor_matrix <- cor(credit_data[, c('Age', 'Income', 'Credit.Score', 'Loan.Amount')], use = "complete.obs")

# Melt the correlation matrix
melted_cor_matrix <- melt(cor_matrix)

# Create the heatmap using ggplot2
ggplot(data = melted_cor_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  geom_text(aes(Var1, Var2, label = round(value, 2)), color = "black", size = 4) +
  labs(title = "Correlation Heatmap", x = "", y = "")

# Plot count of Gender by State
ggplot(credit_data, aes(x = State, fill = Gender)) +
  geom_bar(position = "dodge") +
  labs(title = "Count of Gender by State",
       x = "State",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Income Distribution by Employment Profile
ggplot(credit_data, aes(x = `Employment.Profile`, y = Income)) +
  geom_boxplot() +
  labs(title = "Income Distribution by Employment Profile",
       x = "Employment Profile",
       y = "Income") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Research Question 1: How does income vary across different states?
ggplot(credit_data, aes(x = State, y = Income)) +
  geom_boxplot() +
  labs(title = "Income Distribution across States",
       x = "State",
       y = "Income") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Research Question 2: How does credit score correlate with loan amount?
ggplot(credit_data, aes(x = Credit.Score, y = Loan.Amount)) +
  geom_point() +
  labs(title = "Credit Score vs Loan Amount",
       x = "Credit Score",
       y = "Loan Amount")

# Research Question 3: What is the distribution of profile scores across different occupations?
ggplot(credit_data, aes(x = Occupation, y = Profile.Score)) +
  geom_boxplot() +
  labs(title = "Profile Score Distribution by Occupation",
       x = "Occupation",
       y = "Profile Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#machine learning
#Linear Regression
lm_model <- lm(Credit.Score ~ ., data = credit_data)
summary(lm_model)


# Check the structure and summary of the data
str(credit_data)
summary(credit_data)

# Check for duplicates and missing values
duplicates <- sum(duplicated(credit_data))
missing_values <- colSums(is.na(credit_data))
cat("Duplicates: ", duplicates, "\n")
cat("Missing Values:\n")
print(missing_values)

# Impute missing values for Occupation where Employment Profile is "Unemployed"
credit_data$Occupation[credit_data$"Employment Profile" == "Unemployed" & is.na(credit_data$Occupation)] <- "None"

# Impute remaining missing values in Occupation with a placeholder (e.g., "Unknown")
credit_data$Occupation[is.na(credit_data$Occupation)] <- "Unknown"

# Convert 'Number of Existing Loans' to factor
credit_data$Number.of.Existing.Loans <- as.factor(credit_data$Number.of.Existing.Loans)

# Confirm there are no more missing values
missing_values <- colSums(is.na(credit_data))
cat("Missing Values After Imputation:\n")
print(missing_values)

# Set a seed for reproducibility
set.seed(123)

# Split the data into training and testing sets manually
sample_size <- floor(0.8 * nrow(credit_data))
train_indices <- sample(seq_len(nrow(credit_data)), size = sample_size)
train_data <- credit_data[train_indices, ]
test_data <- credit_data[-train_indices, ]

# Prepare the data for modeling
train_x <- model.matrix(Credit.Score ~ .-1, data = train_data)
train_y <- train_data$Credit.Score
test_x <- model.matrix(Credit.Score ~ .-1, data = test_data)
test_y <- test_data$Credit.Score

# Ridge Regression using cross-validation
ridge_model <- cv.glmnet(train_x, train_y, alpha = 0, standardize = TRUE)

# Predict using the fitted ridge model
ridge_pred <- predict(ridge_model, s = "lambda.min", newx = test_x)

# Calculate RMSE
ridge_rmse <- sqrt(mean((ridge_pred - test_y)^2))
cat("Ridge Regression RMSE: ", ridge_rmse, "\n")

# Optionally, you can plot the cross-validation results
plot(ridge_model)

# Lasso Regression using cross-validation
lasso_model <- cv.glmnet(train_x, train_y, alpha = 1, standardize = TRUE)

# Predict using the fitted lasso model
lasso_pred <- predict(lasso_model, s = "lambda.min", newx = test_x)

# Calculate RMSE for Lasso Regression
lasso_rmse <- sqrt(mean((lasso_pred - test_y)^2))
cat("Lasso Regression RMSE: ", lasso_rmse, "\n")

# Plot the cross-validation results for Lasso Regression
plot(lasso_model)

# Ensure rpart package is loaded for Decision Trees
if (!requireNamespace("rpart", quietly = TRUE)) {
  install.packages("rpart")
}
library(rpart)

# Decision Trees
tree_model <- rpart(Credit.Score ~ ., data = train_data, method = "anova")

# Predict using the decision tree model
tree_pred <- predict(tree_model, newdata = test_data)

# Calculate RMSE for Decision Trees
tree_rmse <- sqrt(mean((tree_pred - test_y)^2))
cat("Decision Tree RMSE: ", tree_rmse, "\n")

# Plot the decision tree (optional, for visualization)
plot(tree_model)
text(tree_model)


