# Titanic Survival Prediction using KNN

A beginner machine learning project that predicts whether a Titanic passenger survived or not using the **K-Nearest Neighbors (KNN)** classification algorithm.

## 📌 Project Overview
 
The goal of this project is to build a KNN classification model using the Titanic dataset and predict passenger survival based on features such as:

- Passenger class
- Sex 
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard 
- Fare
- Port of embarkation

The project focuses on understanding the basic machine learning workflow, including data preprocessing, feature encoding, feature scaling, model training, and prediction.

## 📊 Dataset

The Titanic dataset is obtained using the **Seaborn** library.

The target variable is:

- `survived = 0` → Did not survive
- `survived = 1` → Survived

## ⚙️ Machine Learning Workflow

The following steps were performed:

1. Loaded the Titanic dataset
2. Explored the dataset
3. Checked and handled missing values
4. Removed unnecessary and redundant features
5. Separated features (`X`) and target (`y`)
6. Split the data into training and testing sets
7. Applied One-Hot Encoding to categorical features
8. Applied `StandardScaler` to scale the features
9. Trained a K-Nearest Neighbors classifier
10. Used the trained model to make predictions

## 🧹 Data Preprocessing

### Missing Values

Missing values were handled before training the model.

- Missing `age` values were filled using the median.
- Missing categorical values were filled using the mode.
- Features with excessive missing information were excluded.

### Feature Selection

The following features were used:

- `pclass`
- `sex`
- `age`
- `sibsp`
- `parch`
- `fare`
- `embarked`

Redundant or inappropriate features such as `alive` were removed to avoid data leakage.

### One-Hot Encoding

Categorical features such as `sex` and `embarked` were converted into numerical features using One-Hot Encoding.

### Feature Scaling

`StandardScaler` was used because KNN is a distance-based algorithm.

The scaler was fitted only on the training data and then used to transform both training and testing data.

## 🤖 Model

The machine learning algorithm used is: 

**K-Nearest Neighbors (KNN)**

Initial model:

```python
KNeighborsClassifier(n_neighbors=5)
