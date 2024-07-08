import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn as ib
import warnings
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
df = pd.read_csv("stroke.csv")

# Drop the 'id' column
df = df.drop(['id'], axis=1)

# Replace 'Other' with 'Female' in 'gender' column
df['gender'] = df['gender'].replace('Other', 'Female')

# Display the value counts for 'gender' column
df['gender'].value_counts()

# Convert specific columns to string type
df[['hypertension', 'heart_disease', 'stroke']] = df[['hypertension', 'heart_disease', 'stroke']].astype(str)

# Fill missing values in 'bmi' column with median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Oversample the minority class
oversample = RandomOverSampler(sampling_strategy='minority')
X = df.drop(['stroke_1'], axis=1)
y = df['stroke_1']
X_over, y_over = oversample.fit_resample(X, y)

# Standardize the features
s = StandardScaler()
df[['bmi', 'avg_glucose_level', 'age']] = s.fit_transform(df[['bmi', 'avg_glucose_level', 'age']])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=42)

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Create the KNN classifier object
knn_clf = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
knn_clf.fit(X_train, y_train)

# Perform predictions on the test dataset
y_pred_knn = knn_clf.predict(X_test)

# Printing accuracy of the model
print('Accuracy:', accuracy_score(y_test, y_pred_knn))

# Save the model to a file using pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(knn_clf, f)
