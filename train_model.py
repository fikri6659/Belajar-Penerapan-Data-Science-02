import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
import warnings
import os
os.makedirs('model', exist_ok=True)
warnings.filterwarnings('ignore')

# Set visualisasi
plt.style.use('default')
sns.set_palette('Set2')

# Load dataset
df = pd.read_csv('students_performance.csv')

# Calculate average score
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3

# Calculate median for threshold
median_avg = df['average_score'].median()
threshold = median_avg - 15

# Create dropout risk variable
df['dropout_risk'] = (df['average_score'] < threshold).astype(int)

print(f'Median average score: {median_avg:.2f}')
print(f'Threshold for dropout risk: {threshold:.2f}')
print(f'Number of at-risk students: {df["dropout_risk"].sum()}')
print(f'Percentage of at-risk students: {df["dropout_risk"].mean()*100:.2f}%')

# Encode categorical variables
le = LabelEncoder()

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                    'lunch', 'test preparation course']

for col in categorical_cols:
    df[col + '_encoded'] = le.fit_transform(df[col])

# Create dummy variables for race/ethnicity
df = pd.get_dummies(df, columns=['race/ethnicity'], prefix='race', drop_first=True)

# Print available columns for debugging
print(f'Available columns: {list(df.columns)}')

# Select features - use the actual column names from the dataset
feature_cols = ['gender_encoded', 'parental level of education_encoded', 
                'lunch_encoded', 'test preparation course_encoded',
                'math score', 'reading score', 'writing score',
                'race_group B', 'race_group C', 'race_group D', 'race_group E']

# Check if all race columns exist
race_cols = ['race_group B', 'race_group C', 'race_group D', 'race_group E']
existing_race_cols = [col for col in race_cols if col in df.columns]

if len(existing_race_cols) < len(race_cols):
    print(f'Warning: Some race columns missing. Found: {existing_race_cols}')
    print(f'Available columns: {list(df.columns)}')
    # Use all available race columns
    feature_cols = ['gender_encoded', 'parental level of education_encoded', 
                    'lunch_encoded', 'test preparation course_encoded',
                    'math score', 'reading score', 'writing score'] + existing_race_cols
else:
    feature_cols = ['gender_encoded', 'parental level of education_encoded', 
                    'lunch_encoded', 'test preparation course_encoded',
                    'math score', 'reading score', 'writing score'] + existing_race_cols

X = df[feature_cols]
y = df['dropout_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Train size: {X_train.shape[0]}')
print(f'Test size: {X_test.shape[0]}')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f'\nModel Performance:')
print(f'Accuracy: {accuracy:.4f}')
print(f'AUC: {auc:.4f}')
print(f'\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f'\nConfusion Matrix:')
print(cm)

# Save model and preprocessor

# Save model
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature columns
with open('model/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print('\nModel and preprocessor saved successfully!')

# Save processed dataset for Metabase
df.to_csv('students_performance_processed.csv', index=False)
print('Processed dataset saved successfully!')

# Feature importance
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print('\nFeature Importance:')
print(feature_importance)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy bar chart
accuracies = [accuracy]
axes[0, 0].bar(['Random Forest'], accuracies, color=['#2ecc71'])
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_ylim(0.7, 1.0)

for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# AUC bar chart
aucs = [auc]
axes[0, 1].bar(['Random Forest'], aucs, color=['#3498db'])
axes[0, 1].set_ylabel('AUC Score')
axes[0, 1].set_title('Model AUC Score')
axes[0, 1].set_ylim(0.7, 1.0)

for i, v in enumerate(aucs):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# Confusion Matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xticklabels(['Tidak Berisiko (0)', 'Berisiko (1)'])
axes[1, 0].set_yticklabels(['Tidak Berisiko (0)', 'Berisiko (1)'])

# Feature importance
feature_importance.plot(x='feature', y='importance', kind='bar', ax=axes[1, 1], color='#9b59b6')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Importance')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model/model_analysis.png', dpi=300, bbox_inches='tight')
print('\nModel analysis plot saved as model/model_analysis.png')

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('model/roc_curve.png', dpi=300, bbox_inches='tight')
print('ROC curve plot saved as model/roc_curve.png')
