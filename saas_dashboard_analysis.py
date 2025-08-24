# -*- coding: utf-8 -*-
"""
SaaS Data Analyst Script for Stress Analysis

This script generates key performance indicators (KPIs) and visualizations that
would be valuable for a business dashboard focused on student well-being.

Author: Gemini
Date: 2025-08-24
"""

#############################################################################
# Part 1: Introduction to SaaS Data Analysis                             #
#############################################################################

# Welcome to the SaaS Data Analyst's script! In a SaaS (Software as a Service)
# company, data analysts are crucial for understanding user behavior and
# providing insights that drive business decisions. Here, we'll analyze the
# student stress data from a business perspective.

# ---
# Key SaaS Data Analyst Concepts:
# ---
# 1.  **KPI Tracking:** We'll define and track Key Performance Indicators (KPIs)
#     to monitor student well-being.
# 2.  **User Segmentation:** We'll segment users (students) into different
#     groups to understand how stress affects different demographics.
# 3.  **Dashboarding:** We'll create visualizations that could be used in a
#     dashboard to provide a quick overview of the data.

#############################################################################
# Part 2: Setup and Data Loading                                         #
#############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# Part 3: KPI Tracking and User Segmentation                             #
#############################################################################

# ---
# KPI: Average Stress Level
# ---

avg_stress_level = merged_df['stress_level'].mean()
print(f"Average Stress Level: {avg_stress_level:.2f}")

# ---
# KPI: Stress Level by Gender
# ---

stress_by_gender = merged_df.groupby('Gender')['stress_level'].mean()
print("\nAverage Stress Level by Gender:")
print(stress_by_gender)

# ---
# User Segmentation: High-Stress Students
# ---

high_stress_students = merged_df[merged_df['stress_level'] == 2]

print("\nCharacteristics of High-Stress Students:")
print(high_stress_students.describe())

#############################################################################
# Part 4: Dashboard Visualizations                                       #
#############################################################################

# ---
# Expert SaaS Analyst's Thought:
# ---
# "For a SaaS dashboard, clarity and conciseness are key. We want to present
# the most important information in a way that is easy to understand at a
# glance. We'll use a combination of bar charts, pie charts, and tables to
# achieve this."

# ---
# Stress Level Distribution (Pie Chart)
# ---

stress_counts = merged_df['stress_level'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(stress_counts, labels=['Medium', 'Low', 'High'], autopct='%1.1f%%')
plt.title('Distribution of Stress Levels')
plt.show()

# ---
# Top 5 Stress Factors (Bar Chart)
# ---

stress_factors = [
    'anxiety_level', 'self_esteem', 'depression', 'peer_pressure',
    'academic_performance'
]

stress_factor_corr = merged_df[stress_factors + ['stress_level']].corr()['stress_level']

plt.figure(figsize=(10, 6))
stress_factor_corr.nlargest(5).plot(kind='barh')
plt.title('Top 5 Factors Correlated with Stress')
plt.xlabel('Correlation')
plt.show()

# ---
# Key Metrics Table
# ---

key_metrics = {
    'Metric': ['Avg. Anxiety Level', 'Avg. Self Esteem', 'Avg. Depression'],
    'Value': [
        merged_df['anxiety_level'].mean(),
        merged_df['self_esteem'].mean(),
        merged_df['depression'].mean(),
    ]
}

key_metrics_df = pd.DataFrame(key_metrics)
print("\n--- Key Metrics ---")
print(key_metrics_df)
