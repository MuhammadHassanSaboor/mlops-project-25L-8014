"""
MLOps Assignment 1
House Price Prediction using Machine Learning

Student ID: STUDENT_ID

Pipeline:
1. Load dataset
2. Inspect dataset
3. Remove duplicate rows
4. Separate features and target
5. Remove ID column
6. Identify numerical and categorical features
7. Handle missing values
8. Encode categorical features
9. Scale numerical features
10. Split data into training and testing sets
11. Train Random Forest Regressor
12. Evaluate model
13. Save complete pipeline using pickle
"""

import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

STUDENT_ID = "STUDENT_ID"

DATA_PATH = "data/dataset.csv"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")

TARGET_COLUMN = "SalePrice"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 2. START
# ============================================================

print("=" * 70)
print("HOUSE PRICE PREDICTION - MLOps TRAINING PIPELINE")
print("=" * 70)

print(f"\nStudent ID: {STUDENT_ID}")


# ============================================================
# 3. CHECK DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at: {DATA_PATH}\n"
        "Please make sure your file is named dataset.csv "
        "and placed inside the data/ folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


# ============================================================
# 4. DISPLAY BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 5. CHECK TARGET COLUMN
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found in the dataset."
    )

print(f"\nTarget column: {TARGET_COLUMN}")


# ============================================================
# 6. REMOVE DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: REMOVING DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed.")
else:
    print("No duplicate rows found.")


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: CHECKING MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print(f"\nTotal missing values: {total_missing}")

if total_missing > 0:

    print("\nColumns containing missing values:")

    missing_columns = missing_values[missing_values > 0].sort_values(
        ascending=False
    )

    print(missing_columns)

    print(
        "\nMissing values will be handled automatically "
        "by the preprocessing pipeline."
    )

else:
    print("\nNo missing values found.")


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: SEPARATING FEATURES AND TARGET")
print("=" * 70)

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape  : {y.shape}")


# ============================================================
# 9. REMOVE ID COLUMN
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: REMOVING ID COLUMN")
print("=" * 70)

if "Id" in X.columns:

    X = X.drop(columns=["Id"])

    print("\n'Id' column removed because it is an identifier.")

else:

    print("\n'Id' column was not found.")


# ============================================================
# 10. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: IDENTIFYING FEATURE TYPES")
print("=" * 70)

numerical_columns = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print(f"\nNumber of numerical columns : {len(numerical_columns)}")
print(f"Number of categorical columns: {len(categorical_columns)}")

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 11. NUMERICAL PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: CREATING NUMERICAL PREPROCESSING")
print("=" * 70)

SCALER = StandardScaler()

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
             SCALER
        )
    ]
)

print(
    "\nNumerical preprocessing:"
    "\n1. Missing values -> median"
    "\n2. Features -> StandardScaler"
)


# ============================================================
# 12. CATEGORICAL PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: CREATING CATEGORICAL PREPROCESSING")
print("=" * 70)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

print(
    "\nCategorical preprocessing:"
    "\n1. Missing values -> most frequent value"
    "\n2. Categories -> One-Hot Encoding"
)


# ============================================================
# 13. COMBINE PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 10: COMBINING PREPROCESSING PIPELINES")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

print("\nNumerical and categorical preprocessing combined.")


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 11: TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Test size       : {TEST_SIZE * 100:.0f}%")


# ============================================================
# 15. CREATE RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 12: CREATING MACHINE LEARNING MODEL")
print("=" * 70)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nModel: RandomForestRegressor")
print("Number of trees: 200")
print(f"Random state: {RANDOM_STATE}")


# ============================================================
# 16. CREATE COMPLETE PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("STEP 13: CREATING COMPLETE ML PIPELINE")
print("=" * 70)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

print(
    "\nComplete pipeline created:"
    "\nData"
    "\n  ↓"
    "\nMissing Value Handling"
    "\n  ↓"
    "\nCategorical Encoding"
    "\n  ↓"
    "\nNumerical Scaling"
    "\n  ↓"
    "\nRandom Forest"
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 14: TRAINING MODEL")
print("=" * 70)

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("\nModel training completed successfully.")


# ============================================================
# 18. MAKE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 15: MAKING PREDICTIONS")
print("=" * 70)

y_pred = pipeline.predict(X_test)

print("\nPredictions generated successfully.")


# ============================================================
# 19. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 16: MODEL EVALUATION")
print("=" * 70)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\nEvaluation Results:")
print("-" * 40)

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print("-" * 40)


# ============================================================
# 20. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 17: SAVING MODEL")
print("=" * 70)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )

print(f"\nComplete model pipeline saved successfully.")
print(f"Model path: {MODEL_PATH}")


# ============================================================
# 21. VERIFY MODEL FILE
# ============================================================

if os.path.exists(MODEL_PATH):

    model_size = os.path.getsize(MODEL_PATH)

    print(f"Model file size: {model_size / 1024:.2f} KB")

else:

    raise RuntimeError(
        "Model file was not created successfully."
    )


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"""
Student ID        : {STUDENT_ID}

Dataset           : {DATA_PATH}
Target            : {TARGET_COLUMN}

Total samples     : {len(df)}
Training samples  : {len(X_train)}
Testing samples   : {len(X_test)}

Numerical features: {len(numerical_columns)}
Categorical features: {len(categorical_columns)}

Model             : RandomForestRegressor
Number of trees   : 200

MAE               : {mae:,.2f}
RMSE              : {rmse:,.2f}
R² Score          : {r2:.4f}

Saved model       : {MODEL_PATH}
"""
)

print("=" * 70)
print("Done!")
print("=" * 70)
