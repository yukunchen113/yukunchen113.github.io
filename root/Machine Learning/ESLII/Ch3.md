---
layout: default
title: Ch3
permalink: /ml/ESLII/Ch3/
---
# ESLII Chapter 3: Linear Methods for Regression
## General Tips for Chapter 3:
### Tip #1: Understanding Equations
When looking at equations, it would help to think simple when analyzing equations for an intuitive sense of whats going on. I like to think of N as being equal to 1 unless otherwise stated (or if there is a reason it can't be), where you only have one data point. Also, know what assumptions you can make, and how that translates to math, when trying to solve the equations yourself. There are many subtlties in the wording and equations. For example, if you have $XX^T$ vs $X^TX$, _why do we use one or the other?_ Assuming that X is zero centered, and is a vector, then these would both represent variance. $XX^T$ would represent a matrix of how much each of the features covary (covarince matrix scaled by N). If you took the diagonal of this matrix as a vector, then it would represent a elementwise multiplication of the $X$ vector with itself. $X^TX$ on the other hand, represents sum of the total variance across each feature. In otherwords, it is the sum of the diagonal of the $XX^T$ matrix. You can see how, each of these differs in a practical sense. One gives a detailed breakdown of the variances across each feature, and the other gives an overall view of the variance in the system. 

Make sure that you know the purpose behind each equation, and each operation. Looking at equations as just numbers, variables and functions is like reading a book by just paying attention to each single word, instead of taking in what the overall story is. Instead, keep in mind what each aspect of an equation represents, and what it means to be doing a certain operation. In the future, this will help you translate theory into math, then further on into effcient algorithms. This doesn't mean to neglect proving them yourself, proving them yourself will get you familiar with the tools you can use.

One thing that will help with representations, is by looking at an equation vector-wise. In a $Nxp$ matrix $X$, $N$ signifies the datapoints, $p$ signifies the features. Becareful of which part is multiplying which, sometimes, one matters more than the other. 

### Tip #2: Other General Tips:
- things marked with _An Idea:_ are not a necessity to understand, and is just there for my own notes.
- things marked with _Question:_ are unanswered questions of mine that are about the content.
- __[Code is available! Click here!]({{ site.github.owner_url }}/yukunchen113.github.io/tree/master/root/Machine%20Learning/ESLII/Ch3_code)__ This code is for the various parts in the chapter, and each file will be specified below.

## Chapter 3.1
What does _["E(Y\|X)  linear in the inputs"]({{ site.baseurl }}{% link _posts/2019-01-16-ESLII-Chapter-3-Reference.md %}#linear_in_inputs)_ mean?

Assuming that Y and X have a linear relationship is a direct way to find if they are proportional/correlated. Since this method is simple, and gets the job done to a sufficent degree for many tasks, it is used often.

Since we are making such heavy structural assumptions, which leaves few degrees of freedom that the data has to tune, this model uses less data than non linear models.

## Chapter 3.2 Linear Regression Models and Least Squares
### Introduction
X might not be purely data. It can also be a transformation of your data.

why is (3.2) reasonable only on the assumption that the data is independent, random draws?
- Independent means that the current data drawn isn't affected by any of the previous data drawn, and won't affect any of the next pieces of data drawn. If your data was correlated in this way, the dataset that you have might be biased, depending on the way of how you sampled your data. This will cause your model to be slightly worse as your model will be good at predicting the more biased towards the way you draw your data. 
-If your data wasn't selected randomly, then of course, the true distribution won't be well represented.
- Equation (3.2) is dependent on your data, and is a direct way to form your model towards your data. It doen't build in any assumptions (which could be done though constraints). This means that you want your data to skewed, otherwise, (3.2) won't bea able to help. 
- Your model is trying to model a distribution between the relationship from X to Y. If this distribution was moving/changing given the previous input, (3.2) as a loss function won't be able to minimize it,

The why is it the criterion still valid if the the data is not random and independent? Why does the $y_i$'s have to be conditionally independent given the x?
- Even though your dataset won't be as complete in some areas of the data, it would still be able to predict an approximately correct Y given X in the parts where the data is complete. The relationship between X and Y is not affected, only the samples. 
- The keyword here is _valid_. The model is still valid, though is not statistically reasonable.
- for example, lets say you wanted to measure the temperature of water over time, which is a bunch of data points that you gather. However, your measuring equipement begins to heat up the longer that you use it, causing the water to heat up inadvertently. This means that your data will be skewed as time passes, and your model is wrong.

Here is a very interesting observation. Notice figure 3.2, and how it makes sense. We are using $X$ to construct $\hat{Y}$. $\hat{Y}$ is literally using the $X$ as components. $\hat{Y} = X\hat{\beta}$. But how can be think of this a projection?  
- notice how the equation to calculate the $\hat{\beta}$ is $X(X^TX)^{-1}y$ If we imagine the X and y as vectors, this is _exactly_ the equation for projection! Wen we are constructing the $\hat{\beta}$, we are projecting Y onto the space of X. The representation of this new, projected Y (Known as $\hat{Y}$) inside the X space, is the $\hat{\beta}$. The $\hat{\beta}$ are the sizes of the component basis in the X space to construct the $\hat{Y}$. 
- Now why is this significant? Well, now we know that how linear/additive models estimate E(Y\|X)! They project Y into a space that could be represented by X, and the each of the terms in the additive model are components of this space. After projecting each data point, the resulting components are averaged. This resulting average is what approximates E(Y\|X). 
- We can see that since we are representing Y in our "observable" space (observable through X) which is the best we can get from the data, where is the assumptions coming in to play?
	- The assumptions is that Y can be _linearly_ projected onto the space of X. There might be a non linear transformation, but we have made the assumption that it is a linear one.

This is an interesting problem that will be addressed later, so keep it in mind. If the features are not all linearly independent with one another, Then some of the $\hat{\beta}$ will not be unique. This means that we will not have a unique solution for our training data. This is a problem as some solutions might generalize better than others, and we would like to have the best one.

When deriving some of these equations, we will assume linearity. This is done by letting $y = X\beta +\epsilon$

To get the variance under (3.8):
- We can think of $\hat{sigma}^2$ as the variance across one dimension of y. (Each dimension of y is a data point)
- First, we should see that the error, $\epsilon = y-\hat{y}$ 
	- $y = X\beta +\epsilon$, and $\hat{y} = X\hat{\beta}$.
	- $\hat{\beta}=\beta$ because of our assumption, $E(Y\|X) = X\beta$
		- $E(\hat{\beta})=(X^TX)X^Ty=(X^TX)X^TX\beta=\beta$
- $E(\epsilon^2) = \sigma^2$
- We should also notice that $\hat{y}$ exists in p+1 dimensions as it is constructed from the input space, and is the p+1 dimensional part of y. y is a N dimensional vector. This means the residual which exists in the space that is orthogonal to $\hat{y}$ is a N-(p+1) = N-p-1 dimensional space. 
- Naturally, we divide the sum by N-p-1 to make it unbiased. (each unknown dimension will uniformly contribute to the variance.)

What is the point of finding these? Remember that we are trying to find out the properties of $\hat{$\beta}$. We needed to estimate $\sigma$ since it was in our equation of variance for $\hat{$\beta}$

(N-p-1)$\hat{\sigma}$ is sampled from a chi-squared distribution, as we are naturally adding together the dimensions across $y-\hat{y}$, where the distribution across each of these N-p-1 dimensions is normally distributed. (chi-squared is a distribution of a sum of normals.)

Why is $\hat{\beta}$ is statistically independent with $\hat{\sigma}$? $\hat{\beta}$ exists in the X space of p+1 dimensions, and $\hat{\sigma}$ exists in the orthogonal space of N-p-1 dimensions.

What is the point of finding the distribution of $\hat{\beta}$? We can now do hypothesis tests and confidence intervals to evaluate our $\hat{\beta_j}$.

We want to test $\hat{\beta_j}$ to see if it is 0. This way, we will know if the corresponding input is correlated/contribute to the output.

For (3.12), $v_j$ represents the standard deviation of the jth feature vector. (3.12) is just using the variance that we have calculated above, elementwise.
<!--code this whole process with an example-->

### 3.2.1 Example: Prostate Cancer
<!--code this whole process with an example-->

### 3.2.2 The Gauss-Markov Theorem
Remember that least squares, the linear regression model that we were using is unbiased, as the expected value of the true $y$ is equal to our estimated $\hat{y}$. The bias we are talking about is a mathematical bias. Here, we assume that $y = X\beta$, which corresponds with model. However notice how we are only putting an assumption on the structure of the data. There are many other methods on how we can build in assumptions. For example, we can also assume/bias towards certain kinds of weights, or we can make assumptions on the size of our models. Why would we want to do this? We want to make up for our lack of trainning data through assumptions. This means that we would like to decrease the variance of our model. Methods on how to do will be explained further down. For now we will see how, even though least squares is the unbiased linear estimator with the smallest variance, this variance might still be too high, and we would need to add new, biased methods to compensate.

The gauss markov theorem shows how least squares is has the smallest variance across all unbiased linear estimators.

<!-- should do a study on how much non linear relationships affect linear models.-->
<!-- Prove triangle inequality 3.19 later.-->
For (3.20), they use [bias variance decomposition](https://towardsdatascience.com/mse-and-bias-variance-decomposition-77449dd2ff55) for estimators, Since we are using estimators, we can assume that the true $\theta$ is constant, as we are just estimating given our current data.

for (3.22), we are now dealing with a predictor, which means that we would have to factor in the error of the true model as well now. 

### 3.2.3 Multiple Regression from Simple Univariate Regression
- Multiple linear regression deals with the number of features being greater than 1. (p > 1)
- for the univariate case, (3.24) does make sense, work it out using the sizes of the matrices.

- notice something for (3.26). Remember the equation for projection. This is the exact equation for projection. Regression and projection are the same! As mentioned before. Now the words regress and project will be used interchangably. 

- Though remember, the features that represent our input space (which we use to construct the estimate) might not be orthogonal to each other. This means that the changes in some inputs might affect the $\hat{\beta}$ for other inputs, which might cause some instability. Remember that we are working with just the training set, and instability might hurt generalization.

- (3.27) is considering p=1, where the inputs is centered. This is done to orthogonalize the x from the initial $x_0 = 1$. The residual, z represents the new information that $x_i$ brings in, as we have ortogonalized z with respect to the previous x. The amount of information that the input space contains does not change. The basis is just orthogonal now. 

- the x in (3.27) is a N-dimensional vector representing N different datapoints. 

<!-- program this ortogonalization-->
- why is the regression coefficient of $z_p$ equal to the coefficient on $x_p$?
	- Well, we see that solely $x_p$ contains the information of $z_p$. This means that to include this new information to predict y, we will need the coefficient that corresponds to $z_p$.
	- But, won't the part of $x_p$ which is correlated with the rest of the $x_i$ affect the estimate of y? 
		- Yes, but if we standardize each of the features, then the effect will be small.

- The last sentence on page 54, is trying to say that it doesn't matter how we rearrage our $x_i$, given that we redo the process of regressing. If we have the jth column of x be the last coefficient, then we would need to regress that $x_j$ on all the other $x_i$ that came before it. 




TBD