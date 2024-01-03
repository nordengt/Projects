# 01. Predicting Iris Flower Species

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Algorithms](#technologies)
4. [Dataset](#dataset)
5. [Setup and Usage](#setup-and-usage)

## Introduction:

The Iris Flower Species classification project involves building a machine learning model to identify and classify different species of Iris flowers based on certain features. The Iris dataset is a classic dataset in the field of machine learning and is often used for introductory exercises. It was introduced by the British biologist and statistician Ronald A. Fisher in 1936.

## Problem Definition:

The problem is a multiclass classification task. Given measurements of sepal length, sepal width, petal length, and petal width, the goal is to classify an Iris flower into one of three species: Setosa, Versicolor, or Virginica.

## Algorithms:

Several machine learning algorithms can be applied to solve the Iris Flower Species classification problem. Here are some commonly used algorithms:

- Logistic Regression: A simple linear model that can be used for binary or multiclass classification tasks.
- Decision Trees: These are tree-like structures where each node represents a decision based on one of the input features.
- Random Forest: An ensemble learning method that combines multiple decision trees to improve overall performance.
- Support Vector Machines (SVM): A powerful algorithm for both classification and regression tasks.
- K-Nearest Neighbors (KNN): A simple and effective algorithm that classifies a data point based on the majority class of its k-nearest neighbors.
- Neural Networks: Deep learning models can also be applied, especially for more complex datasets.

## Dataset:

The Iris dataset is a commonly used dataset for this project. It consists of 150 samples of Iris flowers, each from one of three species (Setosa, Versicolor, Virginica). The four features (attributes) measured for each sample are sepal length, sepal width, petal length, and petal width. You can find the Iris dataset in various machine learning libraries like scikit-learn or download it from the UCI Machine Learning Repository.


## Setup and Usage

1. Clone the repository:

```bash
git clone https://github.com/nordengt/PROJNGT.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Navigate to the desired project folder:

```bash
cd A_Classification/01_Predicting_Iris_Flower_Species/
```

4. Open the Jupyter Notebook or Python script where the project code is located:

```bash
jupyter notebook iris_ML.ipynb
```