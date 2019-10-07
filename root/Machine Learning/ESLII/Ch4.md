---
layout: default
title: Ch4
permalink: /ml/ESLII/Ch4/
---
# ESLII Chapter 4: Linear Methods for Classification
## General Tips for Chapter 4:
### Tip #1: Understanding Equations
When looking at equations, it would help to think simple when analyzing equations for an intuitive sense of whats going on. I like to think of N as being equal to 1 unless otherwise stated (or if there is a reason it can't be), where you only have one data point. Also, know what assumptions you can make, and how that translates to math, when trying to solve the equations yourself. There are many subtlties in the wording and equations. For example, if you have $XX^T$ vs $X^TX$, _why do we use one or the other?_ Assuming that X is zero centered, and is a vector, then these would both represent variance. $XX^T$ would represent a matrix of how much each of the features covary (covarince matrix scaled by N). If you took the diagonal of this matrix as a vector, then it would represent a elementwise multiplication of the $X$ vector with itself. $X^TX$ on the other hand, represents sum of the total variance across each feature. In otherwords, it is the sum of the diagonal of the $XX^T$ matrix. You can see how, each of these differs in a practical sense. One gives a detailed breakdown of the variances across each feature, and the other gives an overall view of the variance in the system. 

Make sure that you know the purpose behind each equation, and each operation. Looking at equations as just numbers, variables and functions is like reading a book by just paying attention to each single word, instead of taking in what the overall story is. Instead, keep in mind what each aspect of an equation represents, and what it means to be doing a certain operation. In the future, this will help you translate theory into math, then further on into effcient algorithms. This doesn't mean to neglect proving them yourself, proving them yourself will get you familiar with the tools you can use.

One thing that will help with representations, is by looking at an equation vector-wise. In a $Nxp$ matrix $X$, $N$ signifies the datapoints, $p$ signifies the features. Becareful of which part is multiplying which, sometimes, one matters more than the other. 

### Tip #2: Other General Tips:
- things marked with _An Idea:_ are not a necessity to understand, and is just there for my own notes.
- things marked with _Question:_ are unanswered questions of mine that are about the content.
- __[Code is available! Click here!]({{ site.github.owner_url }}/yukunchen113.github.io/tree/master/root/Machine%20Learning/ESLII/Ch4_code)__ This code is for the various parts in the chapter, and each file will be specified below.

## Chapter 4.1
Main Difference:
- now we predictor, $G(x)$ takes in discrete values

We now divide the input space into regions of the same label, separated by decision boundaries. Decision boundaries is where the prediction of for the ith class equals the prediction for the jth class. $i \neq j, i,j \in 1..K$ where K is the total number of classes

#### Methods for finding linear decision boundaries:
using linear regression (Ch. 2):
- use k classes in y, which will be a k-sized vector.
- see the {x:} equation near the bottom of the page. This shows the decision boundary between the classifications is a hyperplane. This makes sense, as if we take the 2D case of classification, the decision boundary between 2 classes is a line.
- discriminant functions, $\delta_k (x)$ are the predictor functions for each class. We chose the largest value.
- if the predictors are linear in x, why would the decision boundary be as well?
	- being linear in x represents a linear relationship with the inputs, to create the decision boundaries, we just subtract two of these, which is still a linear operation. Therefore, the decision boundaries are still linear in x.
- why is a linear combination of monotonic transformation of the predictors sufficient for the decision boundary to be linear?
	- This should have something to do with the decision boundary being a hyperplane in the input space.
		- It exists in the input space, so is not affect by what we do to the predictors, as long as the transformation of the predictors is monotonic. This means a monotonic transformation in the inputs will have the same effect as the predictors normally.

- for equations 4.1 we see a sigmoid equation which will map the inputs to be between 0 and 1. The equations are just flip of each other. We do this to make the linear equation a probability

- to get equation 4.2, we just applied a log on each of the probabilities in 4.1, and found the difference between them. This should be equated to 0.
	- how does this equation connect with _linear discriminant analysis_ and _logistic regression_?
	- here we are using only one predictor to predict the class, the decision boundary would be f(x)=0.5

- we can create an p dimensional normal representing the component of space orthogonal to the hyperplane, to represent the hyper plane. As well as a cutpoint (a point where the hyperplane converges to a single point, which if remove, will separate the hyperplane into 2.)

- we can change the qualities of the decision boundary by changing our input basis functions. 
	- For example, if we include the squares of the input columns, then the decision boundary will be quadratic.
	- this will also cause the feature space to be larger than p, due to the added basis.
<!-- Program this-->

## Chapter 4.2: Linear Regression of an Indicator Matrix
- Now we are separating the classes. Y is now an NxK indicator response matrix, where for each datapoint, there are K elements, one for each class. 
	- Each of the N rows in Y is a one hot vector, where the 1 represents the activated class. $\hat{\beta}$, which was a K-dimensional vector before, is now a (p+1)xK dimensional matrix, __$\hat{B}$__. X has (p+1) columns with p features, and a 1 as the first column as the bias.
	- the class is the largest index for this prediciton.
- The equation on page 104 is to say that the expected value of $Y_x$ given X = x is supposed to approximate/be equal to the probability of G = k given X = x. So since linear regression aims to estimate the conditional expectation, $E(Y_k\|X = x)$, we can see linear regression as an indirect estimator of the probability. Whether this is a resonable assumption or not remains to be seen.
- the probability is over the column vector. Which means that those of the same class have their own distribution. Though it remains to be seen if they are independent.

- how good is linear regresson for classification/constructing the indicator matrix?
	- In fact, linear regression is rigid, and can allow the estimator to be greater than 1 or negative, both which violate the rules for probability.
	- however, this doesn't mean that linear regression won't be able to be used as a classifier, on some problems, they actually give similar results to other linear methods. I believe that this is because rather than a probability, linear regression will act more like a likelihood.

- (4.5) is using least squares, where the targets are one hot vectors where element k is 1, and the rest are 0, k being the correct class.

- (4.6) is just trying to find the corresponding target that is the minimum to the given prediction. A formal definition for what we were describing above.

- what the first bullet point on page 104 is trying to say, is that we can rearrange the elements in (4.6) and the answer will remain the same. This is because the elements don't depend on eachother at that point in time. There is no positional reliance anymore.
- we can also see that the 4.6 is the same as 4.4. This is because 4.5 caused the responses to be minimized. (So there wouldn't be any response vectors where each element is very large.)

TBD