import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    
    df = pd.concat([X, y], axis=1)
    target_col = y.columns[0]

    # Clean target
    df[target_col] = df[target_col].astype(str).str.strip().str.replace('.', '', regex=False)
    
    # Replace "?" with NaN and drop
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].replace('?', np.nan)
    df = df.dropna().reset_index(drop=True)

    # Drop "fnlwgt" (not useful)
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    # Encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols.remove(target_col)
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    # Encode target
    df[target_col] = LabelEncoder().fit_transform(df[target_col])

    # Scale numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove(target_col)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df.drop(columns=[target_col]), df[target_col]

def train_gmm_classifier(X_train, y_train, n_components=10, covariance_type="full"):
    classes = np.unique(y_train)
    gmms = {}
    for c in classes:
        g = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        g.fit(X_train[y_train == c])
        gmms[c] = g
    return gmms

def gmm_predict(gmms, X):
    classes = sorted(gmms.keys())
    scores = np.zeros((X.shape[0], len(classes)))
    for idx, c in enumerate(classes):
        scores[:, idx] = gmms[c].score_samples(X)
    return np.array(classes)[np.argmax(scores, axis=1)]

def main():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Try different hyperparameters
    best_acc = 0
    best_params = None
    for n in [5, 10, 15, 20]:
        for cov in ["full", "diag", "tied", "spherical"]:
            gmms = train_gmm_classifier(X_train, y_train, n_components=n, covariance_type=cov)
            y_pred = gmm_predict(gmms, X_test)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_params = (n, cov)
            print(f"n={n}, cov={cov}, acc={acc:.4f}")

    print("\nBest params:", best_params, "Accuracy:", best_acc)

    # Retrain with best params
    gmms = train_gmm_classifier(X_train, y_train, n_components=best_params[0], covariance_type=best_params[1])
    y_pred = gmm_predict(gmms, X_test)

    # Final evaluation
    print("\nFinal Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
