# Titanic Survival Prediction

This project uses the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. The dataset contains information about individual passengers such as their age, gender, ticket class, fare, cabin, and whether or not they survived.

## Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

In this project, we build a machine learning model to predict the survival of passengers based on various features available in the dataset. This is a classic beginner project with readily available data, making it an excellent introduction to data science and machine learning.

## Dataset

The dataset used in this project is named `Titanic-Dataset.csv` and contains the following columns:

- `PassengerId`: Unique ID for each passenger
- `Survived`: Survival (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure

- `Titanic-Dataset.csv`: The dataset file.
- `titanic_survival_prediction.py`: The main script for data preprocessing, model training, and evaluation.
- `README.md`: Project documentation.

## Requirements

To run this project, you need the following libraries installed:

- pandas
- numpy
- scikit-learn

You can install the required libraries using pip:

```sh
pip install pandas numpy scikit-learn
```

## Data Preprocessing

1. Drop columns that are not useful for prediction (`PassengerId`, `Name`, `Ticket`, `Cabin`).
2. Handle missing values:
    - Fill missing `Age` values with the median age.
    - Fill missing `Embarked` values with the mode.
3. Convert categorical variables (`Sex`, `Embarked`) to numeric using one-hot encoding.
4. Feature engineering:
    - Create new features `FamilySize` (SibSp + Parch) and `IsAlone` (whether the passenger was alone).

## Model Training and Evaluation

1. Split the dataset into training and testing sets.
2. Standardize the features.
3. Train a Logistic Regression model.
4. Evaluate the model using accuracy, confusion matrix, and classification report.
5. Optionally, try other algorithms like Random Forest and perform hyperparameter tuning.

## Usage

Run the `titanic_survival_prediction.py` script to preprocess the data, train the model, and evaluate its performance.

```sh
python titanic_survival_prediction.py
```

## Results

The Logistic Regression model achieved an accuracy of approximately 81%. Below is the evaluation of the model:

- **Confusion Matrix**:
  ```
  [[90 15]
   [19 55]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             0       0.83      0.86      0.84       105
             1       0.79      0.74      0.76        74

      accuracy                           0.81       179
     macro avg       0.81      0.80      0.80       179
  weighted avg       0.81      0.81      0.81       179
  ```

## Conclusion

This project demonstrates the process of building a machine learning model to predict survival on the Titanic. By preprocessing the data, engineering new features, and experimenting with different algorithms and hyperparameters, we can improve the model's performance.
