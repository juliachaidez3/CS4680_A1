"""
Driver's License Classification
=====================================
Predict if someone qualifies for a driver's license based on their test scores.

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

def load_data():
    """Load the dataset and show basic information"""
    print("Loading dataset...")
    
    # Load the CSV file
    df = pd.read_csv('drivers_license_data.csv')
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target variable distribution:")
    print(df['Qualified'].value_counts())
    
    return df

def prepare_data(df):
    """Clean and prepare the data for machine learning"""
    print("\nPreparing data...")
    
    # Remove the ID column (not needed for prediction)
    df_clean = df.drop('Applicant ID', axis=1)
    
    # Separate features (X) and target (y)
    X = df_clean.drop('Qualified', axis=1)
    y = df_clean['Qualified']
    
    # Convert text categories to numbers
    categorical_columns = ['Gender', 'Age Group', 'Race', 'Training', 'Reactions']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        print(f"  Encoded {col}: {le.classes_}")
    
    print(f"Final features: {list(X.columns)}")
    
    return X, y

def split_data(X, y):
    """Split data into training and testing sets (80% train, 20% test)"""
    print("\nSplitting data...")
    
    # Split the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features so all numbers are in similar ranges"""
    print("\nScaling features...")
    
    # Create a scaler
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully")
    
    return X_train_scaled, X_test_scaled

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train three different machine learning models"""
    print("\nTraining models...")
    
    models = {}
    predictions = {}
    
    # 1. Support Vector Machine (SVM)
    print("  Training SVM...")
    svm = SVC(random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    models['SVM'] = svm
    predictions['SVM'] = svm_pred
    
    # 2. Decision Tree
    print("  Training Decision Tree...")
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    models['Decision Tree'] = tree
    predictions['Decision Tree'] = tree_pred
    
    # 3. Naive Bayes
    print("  Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    nb_pred = nb.predict(X_test_scaled)
    models['Naive Bayes'] = nb
    predictions['Naive Bayes'] = nb_pred
    
    print("All models trained!")
    
    return models, predictions

def evaluate_models(y_test, predictions):
    """Calculate how well each model performed"""
    print("\nEvaluating models...")
    
    results = []
    
    for model_name, y_pred in predictions.items():
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Yes')
        f1 = f1_score(y_test, y_pred, pos_label='Yes')
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'F1-Score': f1
        })
        
        # Print results for this model
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"  F1-Score:  {f1:.3f} ({f1*100:.1f}%)")
    
    return results

def create_visualizations(results):
    """Create simple bar charts to compare model performance"""
    print("\nCreating visualizations...")
    
    # Extract data for plotting
    models = [result['Model'] for result in results]
    accuracy = [result['Accuracy'] for result in results]
    precision = [result['Precision'] for result in results]
    f1_scores = [result['F1-Score'] for result in results]
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy
    axes[0].bar(models, accuracy, color='skyblue', edgecolor='black')
    axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(accuracy):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Precision
    axes[1].bar(models, precision, color='lightgreen', edgecolor='black')
    axes[1].set_title('Model Precision Comparison', fontweight='bold')
    axes[1].set_ylabel('Precision')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(precision):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: F1-Score
    axes[2].bar(models, f1_scores, color='lightcoral', edgecolor='black')
    axes[2].set_title('Model F1-Score Comparison', fontweight='bold')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(f1_scores):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'model_comparison.png'")

def main():
    """Main function that runs the entire analysis"""
    print("=" * 50)
    print("SIMPLE DRIVER'S LICENSE CLASSIFICATION")
    print("=" * 50)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Prepare data
    X, y = prepare_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Step 5: Train models
    models, predictions = train_models(X_train, X_test, y_train, y_test, 
                                     X_train_scaled, X_test_scaled)
    
    # Step 6: Evaluate models
    results = evaluate_models(y_test, predictions)
    
    # Step 7: Create visualizations
    create_visualizations(results)
    
    # Step 8: Final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    # Find best model
    best_model = max(results, key=lambda x: x['Accuracy'])
    print(f"Best model: {best_model['Model']} with {best_model['Accuracy']:.3f} accuracy")
    
    print("\nAll results:")
    for result in results:
        print(f"{result['Model']:15} | Accuracy: {result['Accuracy']:.3f} | "
              f"Precision: {result['Precision']:.3f} | F1: {result['F1-Score']:.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
