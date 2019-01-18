r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Increasing K will may be helpful and after certain point it will make generalization worse. When increasing K to infinty (extremal values for K) all the data point will belong to the same class which is the majority class, this is called underfitting (high bias and low variance) underfitting causes poor performance for our model and thus this won't improve the generalization for the unseen data.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

A matrix $W$ produces the same result as $\alpha W$ for $\alpha > 0$ since effectively score for every class gets multiplied by $\alpha$ and so the class with the largest score remains the same, thus $W$ may have arbitrary constraints so that the values are not too large.
"""

part3_q2 = r"""
**Your answer:**


1. Our linear model tries to find such subspaces of pixels that projections of vectors to those subspaces have high value of projection for true class. We can see some classification errors because some samples have high projection to space of a wrong class which usually happens when a digit resembles another digit.

2. The difference between the models is that KNN tries to generalize data by comparing it to training samples directly while SVM tries to find subspaces of linear space which correspond to different classes. The models are similar because they both try to divide the space into regions which correspond to different classes (but they do it differently).
"""

part3_q3 = r"""
**Your answer:**

1. Learning rate is rather good since the model converges and it converges fast enough. If the learning rate was too low, the model would converge slower and if it was too high, model would diverge or fluctuate.

2. The model is slightly overfitted since it has high accuracy and low loss for training set, and slightly lower accuracy and higher loss for test set.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

Ideal residual plot has all its points at the line $y-\hat{y} = 0$, good model has points close to the line within a small margin.
Our model has a decent fitness since majority of point lie within quite a small margin from the line.
The final set of features is better than top 5 features used separately, residual graphs show that margins for top 5 features are higher than margin for the final set of features.
"""

part4_q2 = r"""
**Your answer:**

1. We need to check orders of magnitude for $\lambda$ rather than exact value since values close to each other produce very similar results, so we use logspace.

2. It was fitted $3 \times 3 \times 20 = 180$ times since it was fitted for 3 folds and for sets of 3 different degrees and 20 different $\lambda$s.
"""

# ==============
