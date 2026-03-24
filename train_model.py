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
df = pd.read_csv('students_performance.csv', sep=';')

print(f'Jumlah baris: {df.shape[0]}')
print(f'Jumlah kolom: {df.shape[1]}')
print(f'\nKolom-kolom: {list(df.columns)}')
print(f'\nTipe data:')
print(df.dtypes)
print(f'\nNilai kosong:')
print(df.isnull().sum())
print(f'\nDistribusi Status:')
print(df['Status'].value_counts())
print(f'\nPersentase Status:')
print(df['Status'].value_counts(normalize=True) * 100)

# Handle missing values - fill with median for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Encode target variable
le_status = LabelEncoder()
df['Status_encoded'] = le_status.fit_transform(df['Status'])

print(f'\nKelas target: {le_status.classes_}')

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['Marital_status', 'Application_mode', 'Course', 'Daytime_evening_attendance', 
                    'Previous_qualification', 'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
                    'Mothers_occupation', 'Fathers_occupation', 'Displaced', 'Educational_special_needs',
                    'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International']

for col in categorical_cols:
    if col in df.columns:
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

# Define feature columns
feature_cols = ['Marital_status_encoded', 'Application_mode_encoded', 'Course_encoded', 
                'Daytime_evening_attendance_encoded', 'Previous_qualification_encoded',
                'Nacionality_encoded', 'Mothers_qualification_encoded', 'Fathers_qualification_encoded',
                'Mothers_occupation_encoded', 'Fathers_occupation_encoded', 'Admission_grade',
                'Displaced_encoded', 'Educational_special_needs_encoded', 'Debtor_encoded',
                'Tuition_fees_up_to_date_encoded', 'Gender_encoded', 'Scholarship_holder_encoded',
                'Age_at_enrollment', 'International_encoded',
                'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
                'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
                'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
                'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
                'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
                'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
                'Unemployment_rate', 'Inflation_rate', 'GDP']

# Check which columns exist
existing_cols = [col for col in feature_cols if col in df.columns]
print(f'\nFeature columns yang tersedia: {len(existing_cols)}')
print(existing_cols)

X = df[existing_cols]
y = df['Status_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'\nTrain size: {X_train.shape[0]}')
print(f'Test size: {X_test.shape[0]}')
print(f'Class distribution in train:')
print(pd.Series(y_train).value_counts())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print(f'\nModel Performance:')
print(f'Accuracy: {accuracy:.4f}')
print(f'AUC: {auc:.4f}')
print(f'\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

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
    pickle.dump(existing_cols, f)

# Save label encoder for target
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le_status, f)

print('\nModel and preprocessor saved successfully!')

# Save processed dataset for Metabase
df.to_csv('students_performance_processed.csv', index=False)
print('Processed dataset saved successfully!')

# Feature importance
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': existing_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print('\nFeature Importance (Top 10):')
print(feature_importance.head(10))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy bar chart
accuracies = [accuracy]
axes[0, 0].bar(['Random Forest'], accuracies, color=['#2ecc71'])
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_ylim(0.5, 1.0)

for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# AUC bar chart
aucs = [auc]
axes[0, 1].bar(['Random Forest'], aucs, color=['#3498db'])
axes[0, 1].set_ylabel('AUC Score')
axes[0, 1].set_title('Model AUC Score')
axes[0, 1].set_ylim(0.5, 1.0)

for i, v in enumerate(aucs):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# Confusion Matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xticklabels(le_status.classes_, rotation=45)
axes[1, 0].set_yticklabels(le_status.classes_, rotation=0)

# Feature importance (top 10)
top_10_features = feature_importance.head(10)
top_10_features.plot(x='feature', y='importance', kind='bar', ax=axes[1, 1], color='#9b59b6')
axes[1, 1].set_title('Feature Importance (Top 10)')
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Importance')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model/model_analysis.png', dpi=300, bbox_inches='tight')
print('\nModel analysis plot saved as model/model_analysis.png')

# ROC Curve (for binary classification, we'll do one-vs-rest)
plt.figure(figsize=(8, 6))
n_classes = len(le_status.classes_)
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=range(n_classes))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'{le_status.classes_[i]} (AUC = {roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i]):.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - One vs Rest')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('model/roc_curve.png', dpi=300, bbox_inches='tight')
print('ROC curve plot saved as model/roc_curve.png')
