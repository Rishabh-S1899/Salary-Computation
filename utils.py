import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- 1. Data Loading and Preprocessing ---

def load_and_clean_data(url):
    """
    Loads and cleans the Adult Income dataset, and reports the number of
    rows dropped due to missing values.
    """
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(
        url, header=None, names=column_names, na_values=' ?',
        sep=r'\s*,\s*', engine='python'
    )

    # 1. Get the number of rows before dropping missing values
    initial_rows = df.shape[0]
    print(f"Initial number of rows loaded: {initial_rows}")

    # 2. Drop the rows with missing values
    df.dropna(inplace=True)

    # 3. Get the number of rows after dropping
    final_rows = df.shape[0]

    # 4. Calculate and print the number of dropped rows
    rows_dropped = initial_rows - final_rows
    print(f"Number of rows with missing values dropped: {rows_dropped}")
    print(f"Final number of rows: {final_rows}")

    # 5. Reset the index and return the cleaned DataFrame
    df.reset_index(drop=True, inplace=True)
    return df

def get_processed_data(df):
    """
    Takes a raw dataframe and returns preprocessed, split, and validated data splits.
    """
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    # Split into training+validation and test sets
    X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Fit preprocessor ONLY on the training+validation data
    X_train_val = preprocessor.fit_transform(X_train_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Split training+validation into final training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val.values, test_size=0.2, random_state=42, stratify=y_train_val.values
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test.values

# --- 2. PyTorch Model Definition ---

class IncomeClassifier(nn.Module):
    """A flexible feed-forward neural network for binary classification."""
    def __init__(self, input_dim, n_hidden1=64, n_hidden2=32, dropout_rate=0.3):
        super(IncomeClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_dim, n_hidden1)
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

# --- 3. Model Training and Hyperparameter Search ---

def train_model_with_history(model, X_train, y_train, X_val, y_val, lr, batch_size, epochs=20):
    """Trains a model and returns a history of training/validation metrics."""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).numpy()
        
        train_acc = correct_train / total_train
        val_acc = accuracy_score(y_val_tensor.numpy(), val_preds)
        
        history['train_loss'].append(total_train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {history["train_loss"][-1]:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {val_acc:.4f}')
        
    return history

def run_hyperparameter_search(param_grid, X_train, y_train, X_val, y_val, epochs=15):
    """Performs a grid search over the specified hyperparameter grid."""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    input_dim = X_train_tensor.shape[1]
    results = []
    
    print("Starting hyperparameter search... 🚀")
    start_time = time.time()

    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            for dropout_rate in param_grid['dropout_rate']:
                print(f"\n--- Training with LR={lr}, Batch Size={batch_size}, Dropout={dropout_rate} ---")
                
                model = IncomeClassifier(input_dim, dropout_rate=dropout_rate)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                
                for epoch in range(epochs):
                    model.train()
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_preds = (torch.sigmoid(val_outputs) > 0.5).numpy()

                val_accuracy = accuracy_score(y_val_tensor.numpy(), val_preds)
                val_f1 = f1_score(y_val_tensor.numpy(), val_preds)
                
                results.append({
                    'lr': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate,
                    'val_accuracy': val_accuracy, 'val_f1_score': val_f1
                })

    end_time = time.time()
    print(f"\nHyperparameter search finished in {end_time - start_time:.2f} seconds. ✨")
    return pd.DataFrame(results)

# --- 4. Final Model Training and Evaluation ---

def train_final_model(best_params, X_train, y_train, X_val, y_val, epochs=20):
    """
    Trains the final model on the combined training and validation sets and returns its history.
    """
    X_full_train = np.concatenate((X_train, X_val), axis=0)
    y_full_train = np.concatenate((y_train, y_val), axis=0)

    X_full_train_tensor = torch.tensor(X_full_train, dtype=torch.float32)
    y_full_train_tensor = torch.tensor(y_full_train, dtype=torch.float32).reshape(-1, 1)

    model = IncomeClassifier(
        input_dim=X_full_train_tensor.shape[1],
        dropout_rate=best_params['dropout_rate']
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    train_dataset = TensorDataset(X_full_train_tensor, y_full_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(best_params['batch_size']), shuffle=True)

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the final model on the test set and plots a confusion matrix.
    """
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_preds = (torch.sigmoid(test_outputs) > 0.5).numpy()

    print("\nFinal Model Performance on Test Set:")
    print(classification_report(y_test_tensor.numpy(), test_preds, target_names=['<=50K', '>50K']))

    cm = confusion_matrix(y_test_tensor.numpy(), test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# --- 5. Plotting ---
def plot_training_curves(history):
    """
    Plots training and validation metrics from a history dictionary.
    Handles cases where validation metrics are not present.
    """
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



def plot_search_results(results_df):
    """Plots the accuracy and F1 score from the hyperparameter search."""
    param_labels = results_df.apply(
        lambda row: f"LR:{row['lr']}, BS:{row['batch_size']}, DO:{row['dropout_rate']}",
        axis=1
    )
    
    plt.figure(figsize=(15, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    sns.lineplot(x=param_labels, y=results_df['val_accuracy'], palette='viridis')
    plt.title('Validation Accuracy for Each Hyperparameter Set')
    plt.xlabel('Hyperparameter Sets')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    # F1 Score Plot
    plt.subplot(1, 2, 2)
    sns.lineplot(x=param_labels, y=results_df['val_f1_score'], palette='plasma')
    plt.title('Validation F1 Score for Each Hyperparameter Set')
    plt.xlabel('Hyperparameter Sets')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()