# Iris Classification Project

## Overview

The Iris Classification Project is a machine learning project that aims to predict the species of an iris flower (Setosa, Versicolor, or Virginica) based on its sepal and petal measurements. This project demonstrates the complete workflow of a classification task — from data exploration to model evaluation and hyperparameter tuning.

The Iris dataset is a classic dataset in machine learning and is widely used for educational purposes due to its simplicity and well-structured features.

---

## Dataset

* **Source:** UCI Machine Learning Repository
* **Number of Samples:** 150
* **Features:**

  * Sepal Length (cm)
  * Sepal Width (cm)
  * Petal Length (cm)
  * Petal Width (cm)
* **Target Variable:** Species (Setosa, Versicolor, Virginica)

---

## Project Workflow

1. **Data Exploration**

   * Checked dataset structure, feature types, and missing values.
   * Used descriptive statistics and visualizations (pairplots, histograms, correlation heatmaps) to understand relationships between features.

2. **Data Preprocessing**

   * Encoded categorical target variable into numeric labels.
   * Standardized features using `StandardScaler` to ensure uniform scale.
   * Removed duplicate rows if present.

3. **Model Training**

   * Trained multiple models including KNN, SVM, Random Forest, Decision Tree, Logistic Regression, Naive Bayes, Gradient Boosting, AdaBoost, and MLP Classifier.
   * Split data into training (80%) and testing (20%) sets.

4. **Model Evaluation**

   * Evaluated models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
   * KNN, SVM, Random Forest, and MLP achieved 100% accuracy.

5. **Hyperparameter Tuning**

   * Used GridSearchCV for models like KNN, SVM, and Random Forest to find optimal parameters.
   * Tuned KNN achieved best parameters: `n_neighbors=9, weights='distance', metric='euclidean'`.

6. **Model Interpretation**

   * Analyzed feature importance (Random Forest) and decision boundaries (SVM/KNN).
   * Identified petal length and petal width as the most important features for classification.

7. **Visualization**

   * Created scatter plots and pairplots to visualize relationships:

     * Sepal Length vs Sepal Width
     * Petal Length vs Petal Width
     * Sepal Width vs Petal Width
     * Sepal Length vs Petal Length
   * Plotted confusion matrices to visualize model performance.

---

## Libraries Used

* `pandas` – Data manipulation
* `numpy` – Numerical computations
* `matplotlib` & `seaborn` – Data visualization
* `scikit-learn` – Machine learning models and evaluation
* `mlxtend` – Optional for decision boundary visualization

---

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/iris-classification.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook `Iris_Classification.ipynb` to execute the full workflow.

---

## Results

* **Best Model:** K-Nearest Neighbors (KNN)
* **Accuracy:** 100%
* **Key Features:** Petal Length and Petal Width
* Confirms that iris species are well-separated in feature space and can be accurately predicted using simple classifiers.

---

## Future Work

* Test the models on larger or more complex datasets.
* Perform feature engineering or dimensionality reduction.
* Explore advanced algorithms like XGBoost or deep learning models.
* Deploy a real-time classification app for iris species.

---

## Author

**MAHALAKSHMI V S**

* Email: [mahalakshmivs1724@gmail.com](mahalakshmivs1724@gmail.com)
* LinkedIn: [https://www.linkedin.com/](https://www.linkedin.com/in/mahalakshmi-vs22/))
