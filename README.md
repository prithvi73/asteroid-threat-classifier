# NASA Asteroid Threat Classifier ☄️

A Supervised Machine Learning classification pipeline that analyzes the orbital mechanics and physical dimensions of Near-Earth Objects (NEOs) to predict whether they are mathematically classified as Potentially Hazardous Asteroids (PHAs).

## Overview
This project processes raw telemetry data sourced from NASA's Jet Propulsion Laboratory (JPL). The pipeline focuses on data cleaning, feature engineering (encoding boolean text data into machine-readable formats), and handling imbalanced datasets where hazardous objects represent a small minority of total observations. 

## Technical Flow
* **Data Ingestion & Cleaning:** Loads a local dataset (`neo_v2.csv`) using Pandas, scrubs missing values, and drops arbitrary string identifiers (like names and orbital body labels) that hold no physical bearing on trajectories.
* **Feature Engineering:** Converts categorical/boolean text formats (e.g., `True/False` threat labels) into absolute integers (`1/0`) for algorithmic processing.
* **Model Training:** Utilizes `train_test_split` to partition the encoded feature matrix, training a `DecisionTreeClassifier` to map the mathematical thresholds for planetary threats.
* **Evaluation & Inference:** Grades model accuracy on a hidden test set and successfully executes real-time inference on high-velocity synthetic asteroid profiles.

## Tech Stack
* Python
* Pandas
* Scikit-learn