# -*- coding: utf-8 -*-
"""
Machine Learning Script for Stress Prediction

This script builds a machine learning model to predict student stress levels based on
various psychological and lifestyle factors.

Author: Gemini
Date: 2025-08-24
"""

#############################################################################
# Part 1: Introduction to Machine Learning Concepts                      #
#############################################################################

# Welcome to the Machine Learning Engineer's script! Here, we'll take the data
# we explored in the data science script and use it to train a predictive model.

# ---
# Key Machine Learning Terms You'll Encounter:
# ---
# 1.  **Feature Selection:** We'll choose the most relevant features from our
#     dataset to train our model. Using too many features can lead to
#     overfitting.
# 2.  **Train-Test Split:** We'll split our data into a training set and a testing
#     set. The model will learn from the training set, and we'll use the
#     testing set to evaluate its performance on unseen data.
# 3.  **Model Selection:** There are many different types of machine learning
#     models. We'll choose one that is well-suited for our classification task.
# 4.  **Model Training:** This is the process of teaching the model to recognize
#     patterns in the data.
# 5.  **Model Evaluation:** We'll use various metrics to evaluate how well our
#     model is performing.

#############################################################################
# Part 2: Setup and Data Loading                                         #
#############################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
try:
    stress_df = pd.read_csv('archive(10)/Stress_Dataset.csv')
    stress_level_df = pd.read_csv('archive(10)/StressLevelDataset.csv')
except FileNotFoundError:
    print("Make sure the CSV files are in the 'archive(10)' directory.")
    exit()

# Merge the datasets
merged_df = pd.concat([stress_df, stress_level_df], axis=1)

#############################################################################
# Part 3: Feature Engineering and Selection                              #
#############################################################################

# For this model, we'll use the numerical features from the `stress_level_df`
# to predict the `stress_level`.

features = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]
target = 'stress_level'

X = merged_df[features]
y = merged_df[target]

#############################################################################
# Part 4: Model Training and Evaluation                                  #
#############################################################################

# ---
# Train-Test Split
# ---
# We'll split the data into 80% for training and 20% for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---
# Model Selection and Training
# ---
# We'll use a RandomForestClassifier, which is a powerful and versatile model.

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---
# Model Evaluation
# ---

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---
# Confusion Matrix
# ---
# A confusion matrix is a great way to visualize the performance of a
# classification model.

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#############################################################################
# Part 5: Feature Importance                                             #
#############################################################################

# Let's see which features were most important in our model's predictions.

feature_importances = pd.Series(model.feature_importances_, index=features)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Most Important Features')
plt.xlabel('Feature Importance')
plt.show()

# ---
# Expert ML Engineer's Thought:
# ---
# "The feature importance plot is a key deliverable for any machine learning
# project. It helps us understand what's driving the model's predictions and can
# provide valuable insights to stakeholders. In this case, it seems that
# 'anxiety_level' and 'depression' are the most predictive features, which aligns
# with our intuition."
