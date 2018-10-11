# Disease Multi-class Classification

X_{train}: 4800*1000, 3 classes


# Methods

## One-vs-One

Train C(C, 2) = C(C-1)/2 classifiers.
E.g. C12, C13, C21 -> (2, 1, 3) -> heuristics to break the tie

## Softmax Regression / Multinomial Logistic Regression
Omitted.


# Difficulties

## Class Imbalance

------
Class | Samples
------
1 | 600
2 | 3600
3 | 600
------

## Confusion Matrix

Simple accuracy is not a good metric for unbalanced dataset.

## Balanced Multi-Class Accuracy

BMCA = 1/C * Sigma\_{i=1 to C}_{TPR\_i}

## Normalized Confusion Matrix

## How to deal with class imbalance?

1. Oversampling/undersampling
2. Change the loss function: add weights

