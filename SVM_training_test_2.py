
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load and preprocess dataset
df = pd.read_csv("test_ds_2.csv")
min_samples_per_class = 3  # Adjust this threshold as needed
# Filter out diseases with very few samples
y_counts = df['Disease'].value_counts()
classes_to_keep = y_counts[y_counts > min_samples_per_class].index
df_filtered = df[df['Disease'].isin(classes_to_keep)]

# Data augmentation: ensure a minimum number of samples per class

for disease in df_filtered['Disease'].unique():
    count = df_filtered['Disease'].value_counts()[disease]
    if count < min_samples_per_class:
        additional_samples_needed = min_samples_per_class - count
        additional_samples = df_filtered[df_filtered['Disease'] == disease].sample(n=additional_samples_needed, replace=True)
        df_filtered = pd.concat([df_filtered, additional_samples], ignore_index=True)

# Prepare features and labels
X = df_filtered[['Symptoms', 'Severity', 'Duration', 'Frequency']]
y = df_filtered['Disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing pipeline
text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'Symptoms'),
        ('categorical', categorical_transformer, ['Severity', 'Duration', 'Frequency'])
    ]
)

# Define the model pipeline with hyperparameter tuning
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', class_weight='balanced'))
])

# Hyperparameter grid for tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']  # Only applies to non-linear kernels
}

# Grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Evaluate model performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Tuned Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print(classification_report(y_test, y_pred))

# Save the best model and the vectorizer
joblib.dump(best_model, 'best_disease_predictor_model_svm.pkl')
# Save the vectorizer from the pipeline
joblib.dump(grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['text'], 'tfidf_vectorizer.pkl')

print("Best model and vectorizer saved successfully!")
'''
output:
Tuned Model Accuracy: 0.9
Confusion Matrix:
 [[3 0 0 ... 0 0 0]
 [0 3 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 4]]
                 precision    recall  f1-score   support

    Alzheimer's       1.00      1.00      1.00         3
         Anemia       1.00      1.00      1.00         3
      Arthritis       1.00      0.67      0.80         3
         Asthma       1.00      0.75      0.86         4
     Bronchitis       1.00      1.00      1.00         3
       COVID-19       1.00      1.00      1.00         4
 Celiac Disease       0.60      1.00      0.75         3
     Chickenpox       0.75      0.75      0.75         4
        Cholera       1.00      1.00      1.00         4
           Cold       0.75      1.00      0.86         3
Common Headache       1.00      1.00      1.00         3
         Dengue       1.00      1.00      1.00         4
       Diabetes       1.00      1.00      1.00         3
       Epilepsy       0.50      0.67      0.57         3
            Flu       1.00      0.67      0.80         3
      Gastritis       1.00      1.00      1.00         4
      Hepatitis       0.75      1.00      0.86         3
   Hypertension       0.67      0.67      0.67         3
Hyperthyroidism       1.00      0.67      0.80         3
 Hypothyroidism       1.00      1.00      1.00         4
       Jaundice       1.00      0.75      0.86         4
          Lupus       0.75      0.75      0.75         4
        Malaria       1.00      0.75      0.86         4
        Measles       0.75      1.00      0.86         3
       Migraine       1.00      1.00      1.00         3
    Parkinson's       1.00      0.67      0.80         3
      Pneumonia       1.00      1.00      1.00         4
      Psoriasis       1.00      1.00      1.00         4
      Sinusitis       0.75      1.00      0.86         3
   Tuberculosis       1.00      1.00      1.00         4
        Typhoid       1.00      1.00      1.00         3
          Ulcer       1.00      1.00      1.00         4

       accuracy                           0.90       110
      macro avg       0.91      0.90      0.90       110
   weighted avg       0.92      0.90      0.90       110

Best model and vectorizer saved successfully!
'''
















'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load and preprocess dataset
df = pd.read_csv("test_ds_2.csv")

# Set minimum samples per class threshold
min_samples_per_class = 3  # Adjust this threshold as needed

# Filter out diseases with very few samples
y_counts = df['Disease'].value_counts()
classes_to_keep = y_counts[y_counts >= min_samples_per_class].index
df_filtered = df[df['Disease'].isin(classes_to_keep)]

# Prepare features and labels
X = df_filtered[['Symptoms', 'Severity', 'Duration', 'Frequency']]
y = df_filtered['Disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing pipeline
text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'Symptoms'),
        ('categorical', categorical_transformer, ['Severity', 'Duration', 'Frequency'])
    ]
)

# Define the model pipeline with hyperparameter tuning
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', class_weight='balanced'))
])

# Hyperparameter grid for tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']  # Only applies to non-linear kernels
}

# Grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Evaluate model performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Tuned Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print(classification_report(y_test, y_pred))

# Save the best model and the vectorizer
joblib.dump(best_model, 'best_disease_predictor_model_svm.pkl')
# Save the vectorizer from the pipeline
joblib.dump(grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['text'], 'tfidf_vectorizer.pkl')

print("Best model and vectorizer saved successfully!")

'''