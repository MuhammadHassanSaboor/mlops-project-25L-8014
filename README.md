# MLOps Project - House Price Prediction

## Project Description

This project is developed for **MLOps Assignment 1**.

The project uses the **Ames Housing Dataset** to predict house sale prices using a `RandomForestRegressor` machine learning model.

The project demonstrates:

- Python and Conda environment management
- Data loading and preprocessing
- Handling missing values
- Numerical feature scaling
- Categorical feature encoding
- Machine learning model training
- Model evaluation
- Saving the trained model using Python `pickle`
- Git and GitHub version control

## Project Structure

```text
mlops-project-25L-8014/
│
├── data/
│   └── dataset.csv
│
├── src/
│   └── train_25L-8014.py
│
├── model/
│   └── trained_model.pkl
│
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** The `data/` and `model/` directories are ignored by Git and are not uploaded to GitHub because they contain the dataset and trained model artifact.

## Requirements

The project uses **Python 3.11** and the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Python's built-in `pickle` module is used to save the trained model, so `pickle` does not need to be installed separately.

## Conda Environment Setup

The Conda environment used for this project is:

```text
mlops_project
```

From the **project root directory**, create the Conda environment:

```bash
conda create -n mlops_project python=3.11
```

Activate the environment:

```bash
conda activate mlops_project
```

## Install Dependencies

After activating the Conda environment, install all required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Training Script

Make sure the Conda environment is activated:

```bash
conda activate mlops_project
```

Run the training script from the **project root directory**:

```bash
python src/train_25L-8014.py
```

## Training Pipeline

The training script performs the following steps:

1. Load the dataset from `data/dataset.csv`
2. Check the dataset information
3. Check for duplicate rows
4. Check for missing values
5. Separate features and target variable
6. Remove the `Id` identifier column
7. Identify numerical and categorical features
8. Handle missing numerical values using the median
9. Handle missing categorical values using the most frequent value
10. Scale numerical features using `StandardScaler`
11. Encode categorical features using `OneHotEncoder`
12. Split the dataset into training and testing sets
13. Train a `RandomForestRegressor`
14. Generate predictions
15. Evaluate the model
16. Save the trained model using `pickle`

## Dataset

The dataset must be placed in:

```text
data/dataset.csv
```

The target column is:

```text
SalePrice
```

The dataset contains:

- **1,460 rows**
- **81 columns**
- **80 input features**
- **1 target variable (`SalePrice`)**

## Data Preprocessing

### Numerical Features

Missing numerical values are handled using:

```text
Median Imputation
```

Numerical features are then standardized using:

```text
StandardScaler
```

### Categorical Features

Missing categorical values are handled using:

```text
Most Frequent Value Imputation
```

Categorical features are then converted into numerical form using:

```text
OneHotEncoder
```

Unknown categories are handled using:

```text
handle_unknown="ignore"
```

## Machine Learning Model

The project uses:

```text
RandomForestRegressor
```

Model configuration:

```text
Number of Trees: 200
Random State: 42
Test Size: 20%
```

The complete preprocessing and machine learning pipeline is saved using Python's `pickle` module.

## Model Output

After successful training, the trained model is saved locally at:

```text
model/trained_model.pkl
```

The saved file contains the complete machine learning pipeline, including preprocessing and the trained Random Forest model.

## Model Evaluation

The model is evaluated using the following metrics:

### MAE

**Mean Absolute Error (MAE)** measures the average absolute difference between the actual and predicted house prices.

### MSE

**Mean Squared Error (MSE)** measures the average squared prediction error.

### RMSE

**Root Mean Squared Error (RMSE)** is the square root of MSE and is expressed in the same units as the target variable.

### R² Score

**R² Score** measures how much of the variation in house prices is explained by the model.

Example results from training:

```text
MAE  : 17,412.83
MSE  : 812,970,963.53
RMSE : 28,512.65
R²   : 0.8940
```

## Complete Installation and Run Commands

All commands below should be executed from the **project root directory**.

### Step 1: Create Conda Environment

```bash
conda create -n mlops_project python=3.11
```

### Step 2: Activate Conda Environment

```bash
conda activate mlops_project
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Training Script

```bash
python src/train_25L-8014.py
```

## Git and Version Control

Git is used to track the project source and configuration files.

The following files and directories are ignored by Git:

```text
data/
model/
venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.pkl
*.h5
*.joblib
.ipynb_checkpoints/
```

The raw dataset and trained model are therefore kept locally and are not uploaded to the GitHub repository.

## Expected Output

After running the training script successfully, the terminal displays:

```text
TRAINING PIPELINE COMPLETED SUCCESSFULLY
```

The model is saved at:

```text
model/trained_model.pkl
```

The terminal also displays the model evaluation results:

```text
MAE
MSE
RMSE
R² Score
```

## Author

**Student NAME:** Muhammad Hassan Saboor

**Student ID:** 25L-8014

**Course:** MLOps for Cloude Native Applications

**Assignment:** Assignment 1 - Git, GitHub and VS Code