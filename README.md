# Regression

A hands-on introduction to how a model learns from data. We start by fitting
a line, then use the same ideas to build a classifier. Every model in this
course, up to transformers, is trained the same way. A function makes a
prediction, a loss measures how wrong it is, and gradient descent updates
the parameters.

- [Linear Regression](./linear-regression.md)
    - learning a function from data
    - mean squared error as a loss
    - gradients and gradient descent
    - training and inference
    - questions on starting points and dataset variability
- [Classification](./classification.md)
    - accuracy and decision boundaries
    - why a line trained with MSE is a poor classifier
    - the sigmoid and predicted probabilities
    - binary cross-entropy
    - training logistic regression
