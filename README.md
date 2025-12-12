

#### This project classifies maternal health risk levels using KNN, MLP, and Random Forest models on the Maternal Health Risk Dataset.

Since all of our code was in the project.ipynb, we've listed the descriptions for each code block in the file.

### Block Descriptions

1. **Imports** – Loads all required libraries for preprocessing, modeling, visualization, and PyTorch-based neural networks.
2. **Load Data** – Reads the Maternal.csv dataset and prints the structure of the features and target variables.
3. **Uniform Label Encoding** – Converts categorical risk labels into numerical format for all models.
4. **Train/Validation/Test Split** – Splits the dataset into consistent training, validation, and test sets across all models.
5. **Global Scaling** – Standardizes all features using training-set statistics to ensure fair comparison across models.
6. **Target Class Distributions** – Plots the class distribution of the risk labels to assess class balance.
7. **Feature Distributions** – Visualizes histograms of all numerical features to understand variable ranges and skewness.
8. **Correlation Heatmap** – Computes and plots feature correlations to identify relationships between predictors.
9. **Evaluation Function** – Defines a reusable function that prints accuracy, precision, recall, F1-score, and confusion matrices.
10. **KNN Cross-Validation** – Performs 5-fold cross-validation over multiple K values to find the optimal number of neighbors.
11. **KNN Training & Evaluation** – Trains KNN with the best K and evaluates it on validation and test sets.
12. **KNN Error Plot** – Visualizes cross-validation training and validation error across different K values.
13. **MLP Dataset Class** – Wraps NumPy data into a PyTorch Dataset for batch loading.
14. **MLP Model Definition** – Implements a multilayer perceptron with batch normalization and dropout.
15. **MLP DataLoaders** – Prepares PyTorch DataLoaders for minibatch training, validation, and testing.
16. **MLP Hyperparameter Search** – Runs a grid search over hidden size, learning rate, and dropout to find the best configuration.
17. **MLP Training Loop** – Trains the MLP for multiple epochs and logs the loss over time.
18. **MLP Validation & Test Evaluation** – Computes predictions and evaluates MLP performance on validation and test sets.
19. **MLP Loss Curve** – Plots the training loss over epochs to visualize optimization behavior.
20. **Random Forest Hyperparameter Search** – Searches over estimators, depth, and split thresholds to find the best RF model.
21. **Random Forest Evaluation** – Evaluates the best Random Forest on validation and test sets using the common evaluation function.
22. **Random Forest Feature Importances** – Plots and prints each feature’s contribution to the Random Forest model.
23. **Model Comparison Setup** – Stores validation and test accuracies for all models and computes their performance differences.
24. **Model Comparison Visualization** – Generates bar and line charts comparing validation vs. test accuracy across the three models.

