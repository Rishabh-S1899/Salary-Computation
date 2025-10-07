import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("KNN CLASSIFIER FOR ADULT INCOME PREDICTION")
print("="*80)
print()

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("📊 STEP 1/6: Loading Dataset...")
print("-"*80)

url = 'https://archive.ics.uci.edu/static/public/2/data.csv'
df = pd.read_csv(url)

print(f"✓ Dataset loaded successfully!")
print(f"  - Total records: {len(df):,}")
print(f"  - Total features: {len(df.columns)}")
print(f"  - Target variable: 'income'")
print()

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("🔧 STEP 2/6: Data Preprocessing...")
print("-"*80)

# Display initial missing values
initial_missing = df.isnull().sum().sum()
print(f"  ⚙ Initial missing values: {initial_missing}")

# Handle missing values
print(f"  ⚙ Handling missing values... ", end="")
df = df.dropna()
print(f"✓ Complete")
print(f"  - Records after cleaning: {len(df):,}")

# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

# Display ORIGINAL class distribution (showing the problem)
print(f"\n  ⚠️  ORIGINAL class distribution (showing 4 classes):")
class_counts_original = y.value_counts()
for class_name, count in class_counts_original.items():
    print(f"    - '{class_name}': {count:,} ({count/len(y)*100:.2f}%)")
print(f"  - Total classes: {len(class_counts_original)}")
print(f"\n  ❌ PROBLEM: Classes with/without dots are identical - just formatting!")
print(f"     Example: '<=50K' and '<=50K.' represent the SAME income level")

# SOLUTION: Merge classes by removing dots
print(f"\n  ✅ SOLUTION: Merging classes by removing formatting dots...")
y = y.str.strip().str.rstrip('.')
print(f"  ✓ Complete")

# Display CLEANED class distribution (2 meaningful classes)
print(f"\n  📊 CLEANED class distribution (2 meaningful classes):")
class_counts_clean = y.value_counts()
for class_name, count in class_counts_clean.items():
    print(f"    - '{class_name}': {count:,} ({count/len(y)*100:.2f}%)")
print(f"  - Total classes: {len(class_counts_clean)}")

# Encode target variable
print(f"\n  ⚙ Encoding target variable... ", end="")
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"✓ Complete")
print(f"  - Final classes: {list(le_target.classes_)}")
print()

# Encode categorical features
print(f"  ⚙ Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for i, col in enumerate(categorical_cols, 1):
    print(f"    [{i}/{len(categorical_cols)}] Encoding '{col}'...", end="\r")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"  ✓ Encoded {len(categorical_cols)} categorical features" + " "*30)
print()

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("✂️  STEP 3/6: Splitting Dataset...")
print("-"*80)

test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
)

print(f"✓ Dataset split completed!")
print(f"  - Training set: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
print(f"  - Testing set: {len(X_test):,} samples ({test_size*100:.0f}%)")
print(f"  - Random state: {random_state}")
print()

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================
print("📏 STEP 4/6: Feature Scaling...")
print("-"*80)

print(f"  ⚙ Applying StandardScaler to normalize features... ", end="")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Complete")
print(f"  - Feature scaling ensures all features contribute equally")
print(f"  - Method: Z-score normalization (mean=0, std=1)")
print()

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING
# ============================================================================
print("🔍 STEP 5/6: Hyperparameter Tuning with GridSearchCV...")
print("-"*80)

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # Power parameter for Minkowski
}

print(f"  📋 Parameter Grid:")
print(f"    - n_neighbors: {param_grid['n_neighbors']}")
print(f"    - weights: {param_grid['weights']}")
print(f"    - metric: {param_grid['metric']}")
print(f"    - p (Minkowski power): {param_grid['p']}")

total_combinations = (len(param_grid['n_neighbors']) * 
                     len(param_grid['weights']) * 
                     len(param_grid['metric']) * 
                     len(param_grid['p']))
print(f"  - Total combinations to test: {total_combinations}")
print()

# Initialize KNN and GridSearchCV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=2,
    n_jobs=-1  # Use all available cores
)

print(f"  🔄 Starting Grid Search with 5-fold Cross-Validation...")
print(f"  (This may take several minutes depending on your system)")
print()

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

print()
print(f"  ✓ Grid Search completed!")
print()
print(f"  🏆 BEST HYPERPARAMETERS FOUND:")
print(f"    - n_neighbors: {grid_search.best_params_['n_neighbors']}")
print(f"    - weights: {grid_search.best_params_['weights']}")
print(f"    - metric: {grid_search.best_params_['metric']}")
print(f"    - p: {grid_search.best_params_['p']}")
print(f"  - Best Cross-Validation Accuracy: {grid_search.best_score_*100:.2f}%")
print()

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
print("📊 STEP 6/6: Model Evaluation on Test Set...")
print("-"*80)

# Get best model
best_knn = grid_search.best_estimator_

# Make predictions
print(f"  ⚙ Making predictions on test set... ", end="")
y_pred = best_knn.predict(X_test_scaled)
print(f"✓ Complete")

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print()
print(f"  🎯 TEST SET ACCURACY: {test_accuracy*100:.2f}%")
print()

# Detailed classification report
print(f"  📈 DETAILED CLASSIFICATION REPORT:")
print("-"*80)
class_names = [str(cls) for cls in le_target.classes_]
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix 
print(f"  📊 CONFUSION MATRIX (2x2):")
print("-"*80)
cm = confusion_matrix(y_test, y_pred)
print(f"\n{'Actual/Predicted':>20} {class_names[0]:>15} {class_names[1]:>15}")
print("-"*80)
for i, cls in enumerate(class_names):
    print(f"{cls:>20} {cm[i][0]:>15} {cm[i][1]:>15}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("🎉 FINAL SUMMARY")
print("="*80)
print(f"✓ Successfully trained KNN model on Adult Income dataset")
print(f"✓ Optimal hyperparameters found through grid search")
print(f"✓ Final test accuracy: {test_accuracy*100:.2f}%")
print()
print(f"📌 KEY INSIGHTS:")
print(f"  - Best K value: {grid_search.best_params_['n_neighbors']} neighbors")
print(f"  - Best weighting: {grid_search.best_params_['weights']}")
print(f"  - Best distance metric: {grid_search.best_params_['metric']}")
print(f"  - Model evaluated on {len(X_test):,} unseen test samples")
print()
print("="*80)
print("✅ PROCESS COMPLETED SUCCESSFULLY!")
print("="*80)