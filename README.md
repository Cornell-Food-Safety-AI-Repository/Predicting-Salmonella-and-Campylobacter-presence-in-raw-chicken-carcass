# RawChickenCarcasses
# Sample Analysis
## Overview
This dataset contains raw chicken carcass sampling data from various poultry establishments across the United States. The data are analyzed for the presence of Salmonella and Campylobacter, critical for monitoring food safety standards. Additionally, the dataset includes detailed weather data corresponding to the collection dates, providing insights into environmental factors that may influence bacterial detection results.


## Data Source
This dataset is provided by the USDA's Food Safety and Inspection Service (FSIS). All data have been collected under strict quality control and assurance procedures to ensure their accuracy and reliability.

## Usage
This dataset is intended for researchers and professionals in food safety, public health monitoring, and environmental science. It allows for the analysis of bacterial contamination in raw chicken and understanding how various environmental factors might impact such results.

# Sample Analysis
## Overview
This Python script facilitates the running of various machine learning models on a specified dataset. It supports logistic regression, MLP classifiers, decision trees, SVM, K-Nearest Neighbors, and Gradient Boosting Machines. Features include handling imbalanced datasets using techniques like Random Over Sampler and SMOTE, configurable model parameters through command-line arguments, and the display of feature importance for applicable models.


## Features
- **Multiple Machine Learning Models:** Choose from several models to train on your data.
- **Configurable Parameters:** Customize model parameters directly via command-line.
- **Feature Importance Display:** For models that support it, display the importance of each feature in the model.
- **Data Preprocessing:** Includes standard scaling and handling missing values.
- **Imbalance Handling:** Options to apply oversampling techniques to balance dataset classes.

## Installation
### Dependencies
To use this script, you'll need Python 3.x and several libraries installed on your system:
- **Pandas**
- **Scikit-learn**
- **Imbalanced-learn**

You can install the necessary Python libraries using pip:
```bash
pip install pandas scikit-learn imbalanced-learn
```
To get started with this project, follow these steps:

```bash
git clone[ https://github.com/FoodDatasets/Predicting-Salmonella-presence-in-raw-chicken-carcass.git
cd Predicting-Salmonella-presence-in-raw-chicken-carcass
```
## Supported Algorithms
-  Logistic Regression
-  Neural Network
-  Decision Tree
-  SVM
-  KNN
-  GBM
## Machine Learning Model Execution Guide
This script allows users to select different machine learning algorithms via command line parameters to train models and evaluate them on a specified dataset.
### Required Arguments
- ` --filepath`: Mandatory. The path to the dataset file.
- ` --model`: Optional. Choose the machine learning model. Default is 'logistic_regression'.
- ` --target`: Optional. Specify the target variable column. Default is 'target'.
### Optional Arguments
- ` --resampling`: Optional. Whether to apply RandomOverSampler for class balancing.
- ` --lr_C`: Optional. Regularization strength for logistic regression (inverse of lambda). Default is 1.0.
- ` --lr_max_iter`: Optional. Maximum iterations for logistic regression. Default is 100.
- ` --mlp_max_iter`: Optional. Maximum iterations for MLP classifier. Default is 200.
- ` --mlp_hidden_layers`: Optional. Number of neurons in the hidden layers for MLP Classifier.
- ` --dt_max_depth`: Optional. Maximum depth for the decision tree. Use 'None' for no limit. Default is 'None'.
- ` --svm_C`: Optional. Regularization parameter for SVM. Default is 1.0.
- ` --svm_kernel`: Optional. Kernel type for SVM. Default is 'rbf'.
- ` --knn_n_neighbors`: Optional. Number of neighbors for KNN. Default is 5.
- ` --gbm_n_estimators`: Optional. Number of boosting stages for GBM. Default is 100.
- ` --gbm_learning_rate`: Optional. Learning rate for GBM.
### Usage Example
Run the script from the command line, specifying the path to your dataset along with options to configure the model:
```bash
python ML_runner.py <path_to_dataset> --model <model_name> --target <target_column> [other options]
```
## Model Performance Results with Resampling Process

The following visualization and tables summarize the performance of different machine learning models after applying resampling.

### Performance Comparison Chart

![Model Performance Comparison](Images/curve_chick.png)

## Confusion Matrices

### Logistic Regression Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|---------------------|---------------------|
| **Actual Negative** | 628                   | 305                   |
| **Actual Positive** | 17                   | 28                   |

### Neural Network Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|---------------------|---------------------|
| **Actual Negative** | 896                   | 37                   |
| **Actual Positive** | 40                   | 5                   |

### Decision Tree Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|---------------------|---------------------|
| **Actual Negative** | 873                   | 60                   |
| **Actual Positive** | 41                   | 4                   |

### SVM Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|---------------------|---------------------|
| **Actual Negative** | 799                   | 134                   |
| **Actual Positive** | 32                   | 13                   |

### KNN Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|---------------------|---------------------|
| **Actual Negative** | 819                   | 114                   |
| **Actual Positive** | 37                   | 8                   |

### GBM Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|---------------------|---------------------|
| **Actual Negative** | 843                   | 90                   |
| **Actual Positive** | 30                   | 15                   |


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# LazyPredict Method
## Dependencies

- Python 3.7+
- pandas
- scikit-learn
- imbalanced-learn
- LazyPredict

``` bash
pip install pandas scikit-learn imbalanced-learn lazypredict
```
## Usage
1. Prepare your dataset file
2. Run the script with the dataset file and target column:
### Command-line Arguments
- `--file_path`: Path to the CSV file containing the dataset (required).
- `--target_label`:SalmonellaSPAnalysis or CampylobacterAnalysis30ml
```bash
python LazyChicken.py /path/to/Dataset_RawChickenCarcasses.xlsx --target SalmonellaSPAnalysis( CampylobacterAnalysis30ml)
```

3. The script will output the number of positive cases for the target variable and evaluate various machine learning models.

## Example Output

### SalmonellaSPAnalysis
#### Model Performance Table

![Model Performance Table](Images/model_performance_table.png)

#### Model Accuracy Comparison

![Model Accuracy Comparison](Images/model_performance_bar.png)

#### Model Comparison

![Model Comparison](Images/output1.png)
### CampylobacterAnalysis30ml
#### Model Performance Table

![Model Performance Table](Images/model_performance_table_C.png)

#### Model Accuracy Comparison

![Model Accuracy Comparison](Images/model_performance_bar_C.png)

#### Model Comparison

![Model Comparison](Images/output2.png)
