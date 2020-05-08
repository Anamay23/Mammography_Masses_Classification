# Mammography_Masses_Classification
This project is a case study comparing the accuracies of different supervised classification algorithms for the Mammographic masses [dataset](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass) provided by [UC, Irvine](https://uci.edu/).

## Problem
Mammography is the most effective method for breast cancer screening available today. However, the low positive predictive value of breast biopsy resulting from mammogram interpretation leads to approximately 70% unnecessary biopsies with benign outcomes. To reduce the high number of unnecessary breast biopsies, several computer-aided diagnosis (CAD) systems have been proposed in the last years. These systems help physicians in their decision to perform a breast biopsy on a suspicious lesion seen in a mammogram or to perform a short term follow-up examination instead.

## Data set
The dataset contains 961 instances of masses detected in mammograms, and contains the following attributes:
* BI-RADS assessment: 1 to 5 (ordinal)
* Age: Patient's age in years (integer)
* Shape: Mass shape: (round = 1, oval = 2, lobular = 3, irregular = 4) (nominal)
* Margin: Mass Margin: (circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5) (nominal)
* Density: Mass Density: (high=1 iso=2 low=3 fat-containing=4) (ordinal)
* Severity: benign=0 or malignant=1 (binominal)

Ignoring BI-RADS as it is _NOT_ a predictive attribute

## Objective
1. To clean and pre-process the data for any missing/spurious values.
2. To predict whether the mass is malignant or benign using different suprevised classification techniques.
3. To see which one yields the **HIGHEST** accuracy as measured with K-Fold cross validation. (K = best value found after trial and error)

## Results
After running various classifiers, the accuracies are as follows:
* Decision Trees - 76.12%
* Random Forest - 77.25%
* SVM (linear kernel) - 79.87%
* SVM (polynomial kernel) - 79.03%
* SVM (RBF kernel) - 80.34%
* K Nearest Neighbours - 80.02% (for k = 7) tested upto k = 50
* Naive Bayes - 78.31%
* Logistic Regression - 80.61%
* Artificial Neural Network (Keras) - 80.36% (for 10 epochs)

Most of the classifiers have an accuracy between 78-80% with Decision Trees and Random Classifier being the outliers.
The classifier with the highest accuracy is **Logistic Regresssion**. 


## References
This case study is part of the final project for [this](https://www.udemy.com/data-science-and-machine-learning-with-python-hands-on/) Udemy course.

