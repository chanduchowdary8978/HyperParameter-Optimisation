# Hyperparameter Tuning Web App (Streamlit)

A **comprehensive Streamlit-based Machine Learning playground** that allows users to upload datasets, select models, tune hyperparameters interactively, and visualize results from a single user interface.

GitHub Repository:
[https://github.com/chanduchowdary8978/HyperParameter-Optimisation](https://github.com/chanduchowdary8978/HyperParameter-Optimisation)

---

## Features

* Upload CSV datasets
* Select dependent and independent variables
* Automatic scatter plot visualizations
* Supports multiple machine learning models
* Full hyperparameter control through sidebar
* Model training and evaluation
* Manifold learning and clustering visualizations

---

## Supported Algorithms

### Regression / Classification

* Elastic Net
* Random Forest (Classifier and Regressor)
* Quantile Regression
* SVM (LinearSVC)
* K-Nearest Neighbours
* Decision Trees
* Gradient Boosting
* PLS Regression (Cross Decomposition)

### Manifold Learning

* Isomap
* Locally Linear Embedding (LLE)
* Spectral Embedding
* t-SNE

### Clustering

* K-Means
* Affinity Propagation
* Mean Shift
* Spectral Clustering
* Agglomerative Clustering
* DBSCAN
* HDBSCAN

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* PyForest

---

## Installation

```bash
git clone https://github.com/chanduchowdary8978/HyperParameter-Optimisation.git
cd HyperParameter-Optimisation
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Dataset Format

* CSV file
* Both numerical and categorical columns supported
* Target variable selected manually by the user

---

## Notes

* Accuracy is used for classification models
* Mean Squared Error or Quantile Loss is used for regression models
* Dimensionality reduction and clustering outputs are visualized in 2D

---

## Future Improvements

* Train/Test split and cross-validation
* AutoML integration
* Model comparison dashboard
* Export trained models
* Dataset preprocessing pipeline

##
