# -*- coding: utf-8 -*-
"""
Data Developer Utils for Stress Data

This script provides a set of reusable classes and functions for handling the
student stress data in a production environment.

Author: Gemini
Date: 2025-08-24
"""

#############################################################################
# Part 1: Introduction to Data-focused Development                       #
#############################################################################

# Welcome to the Data-focused Developer's script! A data developer (or data
# engineer) is responsible for building and maintaining the infrastructure that
# allows data scientists and analysts to do their work. This means writing
# clean, efficient, and reusable code.

# ---
# Key Data Developer Concepts:
# ---
# 1.  **Modularity:** We'll create a class to encapsulate the data loading and
#     cleaning logic, making it easy to reuse in other scripts.
# 2.  **Data Validation:** We'll add checks to ensure the data is in the correct
#     format and within the expected range of values.
# 3.  **Data Transformation:** We'll create functions to transform the data into
#     different formats, such as JSON, which is commonly used for APIs.

#############################################################################
# Part 2: Data Handling Class                                            #
#############################################################################

import pandas as pd
import json

class StressDataHandler:
    """A class to handle loading, cleaning, and transforming the stress data."""

    def __init__(self, stress_path, stress_level_path):
        """Initializes the data handler with the paths to the datasets."""
        self.stress_path = stress_path
        self.stress_level_path = stress_level_path
        self.df = self._load_and_merge_data()

    def _load_and_merge_data(self):
        """Loads and merges the two datasets."""
        try:
            stress_df = pd.read_csv(self.stress_path)
            stress_level_df = pd.read_csv(self.stress_level_path)
        except FileNotFoundError:
            print("Make sure the CSV files are in the correct directory.")
            return None

        return pd.concat([stress_df, stress_level_df], axis=1)

    def validate_data(self):
        """Validates the data to ensure it meets the expected format."""
        if self.df is None:
            return False

        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            print("Error: Missing values found in the data.")
            return False

        # Check for expected columns
        expected_columns = ['Gender', 'Age', 'stress_level']
        if not all(col in self.df.columns for col in expected_columns):
            print("Error: Missing expected columns.")
            return False

        print("Data validation successful.")
        return True

    def get_data_as_json(self, orient='records'):
        """Returns the data as a JSON string."""
        if self.df is None:
            return None

        return self.df.to_json(orient=orient, indent=4)

#############################################################################
# Part 3: Example Usage                                                  #
#############################################################################

if __name__ == "__main__":
    # Create a data handler object
    data_handler = StressDataHandler(
        'archive(10)/Stress_Dataset.csv',
        'archive(10)/StressLevelDataset.csv'
    )

    # Validate the data
    if data_handler.validate_data():
        # Get the data as JSON
        json_data = data_handler.get_data_as_json()

        # Save the JSON data to a file
        with open('stress_data.json', 'w') as f:
            f.write(json_data)

        print("\nSuccessfully saved data to stress_data.json")
