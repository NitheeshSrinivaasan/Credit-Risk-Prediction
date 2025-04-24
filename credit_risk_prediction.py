
# Phase 1: Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import shap
import os

# Ensure required directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/german_credit_data.csv")

# Phase 2: Data Cleaning & EDA
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

print(df.info())
print(df.describe())
print(df.isnull().sum())

# Fill missing values
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# Outlier handling functions
def count_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers)

def cap_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])

outlier_counts = {}
for col in ['Age', 'Credit amount', 'Duration']:
    count = count_outliers(col)
    outlier_counts[col] = count
    cap_outliers(col)

print("\nOutlier Summary:")
for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers")
print(f"Total outliers: {sum(outlier_counts.values())}\n")

# Feature Relationships Analysis
sns.scatterplot(data=df, x="Credit amount", y="Duration", hue="Risk")
plt.title("Credit Amount vs Duration by Risk")
plt.savefig("outputs/rel_credit_duration.png")
plt.show()

sns.scatterplot(data=df, x="Age", y="Credit amount", hue="Risk")
plt.title("Age vs Credit Amount by Risk")
plt.savefig("outputs/rel_age_credit.png")
plt.show()

sns.boxplot(data=df, x="Risk", y="Saving accounts")
plt.title("Saving Accounts vs Risk")
plt.savefig("outputs/rel_saving_risk.png")
plt.show()

sns.boxplot(data=df, x="Risk", y="Checking account")
plt.title("Checking Account vs Risk")
plt.savefig("outputs/rel_checking_risk.png")
plt.show()

sns.countplot(data=df, x="Purpose", hue="Risk")
plt.title("Purpose vs Risk")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/rel_purpose_risk.png")
plt.show()

sns.countplot(data=df, x="Housing", hue="Risk")
plt.title("Housing vs Risk")
plt.tight_layout()
plt.savefig("outputs/rel_housing_risk.png")
plt.show()

sns.boxplot(data=df, x="Risk", y="Age")
plt.title("Age vs Risk")
plt.savefig("outputs/rel_age_risk.png")
plt.show()

# Phase 3: Encoding & Scaling
le = LabelEncoder()
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Risk', axis=1)
y = df['Risk'].apply(lambda x: 1 if x == 'good' else 0)

joblib.dump(X.columns.tolist(), "model/features.pkl")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Phase 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate and plot models
def evaluate_and_plot_models(models, X_train, X_test, y_train, y_test, save_path="outputs/roc_comparison.png"):
    results = []
    plt.figure(figsize=(10, 6))

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

        print(classification_report(y_test, preds))
        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix:\n", cm)
        print("True Positives:", cm[1][1])
        print("True Negatives:", cm[0][0])
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)
        print("ROC AUC Score:", auc)

        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "AUC": auc})

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    results_df = pd.DataFrame(results)
    print("\nModel Comparison Summary:")
    print(results_df.sort_values(by="AUC", ascending=False))
    return results_df

# Phase 5: Model Training, Evaluation & Comparison
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(verbosity=0, eval_metric='logloss'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True)
}

results_df = evaluate_and_plot_models(models, X_train, X_test, y_train, y_test)

# ✅ Phase 6: Save the Best Model (based on AUC)
best_model_name = results_df.sort_values(by="AUC", ascending=False).iloc[0]['Model']
best_model_accuracy = results_df.sort_values(by="AUC", ascending=False).iloc[0]['Accuracy']

print(f"\nBest Model Based on AUC: {best_model_name}")
print(f"Accuracy of the Best Model: {best_model_accuracy:.4f}")

best_model = models[best_model_name]
joblib.dump(best_model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# 🔍 Phase 7: Feature Importance and SHAP Analysis
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.show()

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png")
    plt.close()