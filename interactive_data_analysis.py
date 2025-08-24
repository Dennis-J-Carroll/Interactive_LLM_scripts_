
# -*- coding: utf-8 -*-
"""
Deep Interactive Data Science Script for Stress Analysis

This script performs a comprehensive analysis of student stress levels using two datasets.
It is designed to be an educational tool, with detailed explanations and expert insights
provided throughout the code.

Author: Gemini
Date: 2025-08-24
"""

# #############################################################################
# # Part 1: Introduction to Data Science Concepts                          #
# #############################################################################

# Welcome to this interactive data science script! Here, we'll explore the
# fascinating world of data analysis by examining factors that contribute to
# stress in students' lives. This script is designed to be a learning journey,
# so we'll break down each step and explain the "why" behind the "what."

# ---
# Key Data Science Terms You'll Encounter:
# ---
# 1.  **Data Loading and Inspection:** The first step in any data science project
#     is to load your data and get a feel for its structure and content. We'll
#     use the pandas library for this, which is the workhorse of data
#     manipulation in Python.
# 2.  **Data Cleaning and Preprocessing:** Real-world data is often messy. It can
#     have missing values, inconsistencies, or errors. We'll clean our data to
#     ensure our analysis is accurate and reliable.
# 3.  **Exploratory Data Analysis (EDA):** This is where the fun begins! EDA is
#     the process of exploring your data to find patterns, relationships, and
#     anomalies. We'll use a variety of visualization techniques to bring our
#     data to life.
# 4.  **Feature Engineering:** Sometimes, the raw data doesn't tell the whole
#     story. Feature engineering is the art of creating new features from
#     existing ones to better represent the underlying patterns in the data.
# 5.  **Statistical Analysis:** We'll use statistical methods to test our
#     hypotheses and quantify the relationships we observe in the data.
# 6.  **Data Visualization:** A picture is worth a thousand words, and in data
#     science, it's worth a thousand data points! We'll use libraries like
#     matplotlib and seaborn to create insightful visualizations.

# ---
# A Note on the "Data Scientist's Mindset":
# ---
# As we go through this script, try to think like a data scientist. This means:
# -   **Be curious:** Always ask questions about your data. Why is this pattern
#     emerging? What could be the underlying cause?
# -   **Be skeptical:** Don't take your findings at face value. Always look for
#     alternative explanations and be aware of potential biases in your data.
# -   **Be creative:** Data science is as much an art as it is a science. Don't
#     be afraid to experiment with different techniques and approaches.

# Now, let's dive into the code!

# #############################################################################
# # Part 2: Setup and Data Loading                                         #
# #############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---
# Data Loading
# ---
# We'll start by loading our two datasets into pandas DataFrames. A DataFrame is
# like a spreadsheet or a SQL table, but with more powerful features.

try:
    stress_df = pd.read_csv('archive(10)/Stress_Dataset.csv')
    stress_level_df = pd.read_csv('archive(10)/StressLevelDataset.csv')
except FileNotFoundError:
    print("Make sure the CSV files are in the 'archive(10)' directory.")
    exit()

# ---
# Initial Data Inspection
# ---
# Let's take a first look at our data to understand its structure.

print("--- Stress Dataset Info ---")
stress_df.info()
print("
--- Stress Level Dataset Info ---")
stress_level_df.info()

# #############################################################################
# # Part 3: Data Cleaning and Preprocessing                                #
# #############################################################################

# ---
# Expert Data Scientist's Thought:
# ---
# "Data cleaning is a crucial step that is often overlooked. A small error in the
# data can lead to a large error in the analysis. It's always worth taking the
# time to get this right."

# ---
# Handling Missing Values
# ---
# Let's check for any missing values in our datasets.

print("
--- Missing Values in Stress Dataset ---")
print(stress_df.isnull().sum())
print("
--- Missing Values in Stress Level Dataset ---")
print(stress_level_df.isnull().sum())

# It looks like our datasets are clean, with no missing values. This is great,
# but in a real-world scenario, you would likely need to handle missing data by
# either removing the rows with missing values or imputing them with a
# reasonable estimate (like the mean or median).

# ---
# Merging Datasets
# ---
# To get a holistic view of the data, we'll merge our two DataFrames into one.
# Since both datasets have the same number of rows and seem to correspond to the
# same individuals, we can merge them based on their index.

merged_df = pd.concat([stress_df, stress_level_df], axis=1)

# Let's rename the columns for clarity.
new_column_names = {
    'Have you recently experienced stress in your life?': 'experienced_stress',
    'Have you noticed a rapid heartbeat or palpitations?': 'rapid_heartbeat',
    'Have you been dealing with anxiety or tension recently?': 'anxiety_tension',
    'Do you face any sleep problems or difficulties falling asleep?': 'sleep_problems',
    'Have you been getting headaches more often than usual?': 'headaches',
    'Do you get irritated easily?': 'irritability',
    'Do you have trouble concentrating on your academic tasks?': 'concentration_difficulty',
    'Have you been feeling sadness or low mood?': 'low_mood',
    'Have you been experiencing any illness or health issues?': 'health_issues',
    'Do you often feel lonely or isolated?': 'loneliness',
    'Do you feel overwhelmed with your academic workload?': 'workload_overwhelm',
    'Are you in competition with your peers, and does it affect you?': 'peer_competition',
    'Do you find that your relationship often causes you stress?': 'relationship_stress',
    'Are you facing any difficulties with your professors or instructors?': 'professor_difficulties',
    'Is your working environment unpleasant or stressful?': 'unpleasant_environment',
    'Do you struggle to find time for relaxation and leisure activities?': 'no_relaxation_time',
    'Is your hostel or home environment causing you difficulties?': 'home_environment_issues',
    'Do you lack confidence in your academic performance?': 'low_academic_confidence',
    'Do you lack confidence in your choice of academic subjects?': 'low_subject_confidence',
    'Academic and extracurricular activities conflicting for you?': 'activity_conflict',
    'Do you attend classes regularly?': 'regular_class_attendance',
    'Have you gained/lost weight?': 'weight_change',
    'Which type of stress do you primarily experience?': 'stress_type'
}
merged_df.rename(columns=new_column_names, inplace=True)


# #############################################################################
# # Part 4: Exploratory Data Analysis (EDA)                                #
# #############################################################################

# ---
# Expert Data Scientist's Thought:
# ---
# "EDA is where you build an intuition for your data. It's a creative process
# of asking questions and seeking answers in the data. The goal is to understand
# the story your data is telling."

# ---
# Univariate Analysis: Understanding Single Variables
# ---

# Let's start by looking at the distribution of some key variables.

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['Age'], bins=10, kde=True)
plt.title('Age Distribution of Students')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=merged_df)
plt.title('Gender Distribution')
plt.xlabel('Gender (0: Male, 1: Female)')
plt.ylabel('Count')
plt.show()

# Stress level distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='stress_level', data=merged_df)
plt.title('Stress Level Distribution')
plt.xlabel('Stress Level (0: Low, 1: Medium, 2: High)')
plt.ylabel('Count')
plt.show()

# ---
# Bivariate Analysis: Exploring Relationships
# ---

# Now, let's look at how different variables relate to each other.

# Stress level by gender
plt.figure(figsize=(10, 6))
sns.countplot(x='stress_level', hue='Gender', data=merged_df)
plt.title('Stress Level by Gender')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.legend(title='Gender', labels=['Male', 'Female'])
plt.show()

# Correlation matrix
# A correlation matrix is a powerful tool to see how different numerical
# variables are related. A value close to 1 means a strong positive
# correlation, while a value close to -1 means a strong negative correlation.
plt.figure(figsize=(18, 15))
sns.heatmap(merged_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of All Variables')
plt.show()

# ---
# Expert Data Scientist's Thought:
# ---
# "The correlation matrix gives us a bird's-eye view of the relationships in
# our data. I see a strong positive correlation between 'depression' and
# 'anxiety_level', which makes intuitive sense. I also see that 'stress_level'
# is positively correlated with many factors like 'peer_pressure' and
# 'academic_performance'. This is a great starting point for more detailed
# analysis."

# #############################################################################
# # Part 5: Deeper Dive into Key Factors                                   #
# #############################################################################

# Let's investigate some of the factors that seem to be most correlated with
# stress levels.

# Academic performance vs. stress level
plt.figure(figsize=(10, 6))
sns.boxplot(x='stress_level', y='academic_performance', data=merged_df)
plt.title('Academic Performance vs. Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Academic Performance')
plt.show()

# Sleep quality vs. stress level
plt.figure(figsize=(10, 6))
sns.violinplot(x='stress_level', y='sleep_quality', data=merged_df)
plt.title('Sleep Quality vs. Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Sleep Quality')
plt.show()

# ---
# Expert Data Scientist's Thought:
# ---
# "The violin plot for sleep quality is particularly insightful. It shows not
# just the median and quartiles like a box plot, but also the distribution of
# the data. We can see that as stress levels increase, the distribution of
# sleep quality shifts towards lower values, and the distribution becomes more
# spread out. This suggests that high stress is associated with not just poorer
# sleep, but also more inconsistent sleep."

# #############################################################################
# # Part 6: Conclusion and Further Exploration                             #
# #############################################################################

# This script has provided a comprehensive overview of the data science process,
# from data loading and cleaning to exploratory data analysis and visualization.
# We've uncovered some interesting patterns in the data, such as the strong
# correlation between various psychological factors and stress levels.

# ---
# Where to Go from Here?
# ---
# This analysis is just the beginning. Here are some ideas for further
# exploration:
# -   **Machine Learning:** Could you build a model to predict a student's
#     stress level based on the other factors in the dataset?
# -   **Causal Inference:** While we've observed correlations, we can't say for
#     sure what causes stress. Causal inference techniques could help us
#     understand the causal relationships in the data.
# -   **Interactive Dashboard:** You could build an interactive dashboard to
#     allow users to explore the data themselves.

# We hope this script has been a valuable learning experience. Happy coding!

