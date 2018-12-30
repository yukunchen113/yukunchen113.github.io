---
layout: default
title: Ch2
permalink: /ml/ESLII/Ch2
---
# ESLII Chapter 2: Overview of Supervised Learning
<!-- Introduction/overview of chapter -->

<!-- General Tips
- think simple when looking at equations (N=1)
- use the dimensions of the matrix to your advantage to see how matricies are being manipulated (p roughly represents the features, N represents the number of data points.)
- Bookmark pg 2 for it's equations, I found it helpful to look back on this in future chapters. (You don't want to be deriving it every single time)
- things marked with An Idea: are not a necessity to understand, and is just there for my own notes, as any new ideas happen.
- thigh marked with Question: are unanswered questions of mine that are about the content.
-->

## Chapter 2.3:

### Introduction

<a id = "structural_assumption"></a>
- structural assumptions: 
	- assumptions made about the relationship between all x and y.

- linear models assume linearity in the [structure](#structural_assumption) of the data, so they might be inaccurate if the data is non linear.

- kNN does not give any strong assumptions. What does "strong" mean? While linearity assumes that the mapping from X to Y is linear, which is a global assumption. kNN is more of a local assumption that relies heavily on the training data in a local area. But because of this heavy reliance on the training data, with the addition of a new data point, the model can change significantly. This is bad as then we would have to select our training data very carefully.

### Section 2.3.1: Linear Models and Least Squares
- what is an [affine set]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#affine_set)?

- Analysis on the (p+1) dimensional input-output space:
	- ($X, \hat{Y}$) is represented in a p dimensional space, where each dimension is a feature. Remember that $\hat{Y}$ is just a linear combination of the features in $X$, our estimated value. 
	- the last dimension of (p+1) dimensions, where p of them are from ($X,\hat{Y}$), represents the combination of the uncertain/unknown components in $Y$ (the actual output). These are the components not represented by the hyperplane/features of $X$. This can be seen as if we include a component $\hat{\beta_0}$ in X, this beta is not an uncertainty anymore, and becomes a component of $X$, rather than a bias for Y.
	- since features are not necessarily orthogonal to each other, though are still used as the basis of the space, the input-output space doesn't have to be orthogonal.
- the small $x_i$ in (2.3) represents a data vector of size p.
- [transpose identities](https://math.vanderbilt.edu/sapirmv/msapir/prtranspose.html) help with the proof.
- for information on matrix differentiation see [this](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf), 5. Matrix Differentiation for identities and their proofs (2.4) -> (2.5).
- what does it mean for $X^TX$ to be singular? When does this happen?
	- $X^TX$ represents a matrix whose elements are dot products between each feature vector of size, p. (multiplications between the features in a given datapoint, summed across diffent datapoints), like an uncentered covariance between features.
	- for a row to be 0, (causing a singular matrix) a feature must be orthogonal to every feature (including itself!), so this means that the matrix will be singular if all the elements in a feature vector is 0. So across all data points, a feature is always 0.
- if the mixture of gaussians is not tightly clustered, we can use a random variable to determine which gaussian in the mixture to sample from, then, sample from that gaussian. 

Code for Scenario 1 and Scenario 2, as well as gaussian mixtures is available. Implemented with Python and Numpy.

### Section 2.3.2: Nearest Neighbor Methods
- _voronoi tessellation_ or _voronoi diagrams_ are the splitting of space into different regions.
- becareful of overfitting (k=1), use a test set to determine if your model holds up.
- the _effective_ number of parameter for kNN is $N/k$, which is a rough measure that is proportional to the max number of regions (number of neighborhoods, if the neighborhoods dodn't overlap)
- kNN is good for scenario 2 as scenario 2 has many small regions. As kNN is inherently regional, it will perform well with scenario 2, given a proper k value, even though it has very noisy boundaries. 

### Section 2.3.3: From Least Squares to Nearest Neighbors
<!-- build this program (pg 16, revealing the oracle) -->
- Basis expansion is just a transformation on the inputs. Eg. using $X^2$ rather than just $X$

- Projection pursuit projects your high dimensional data onto a lower dimension, Trying to keep the most informative/interesting parts. Eg. PCA. 

## Chapter 2.4: Statistical Decision Theory
- _minimizing pointwise_ means, minimizing given the samples that we have, as opposed to the whole X distribution (which we don't have).
- we can replace $f(X)$ with $c$ since $f(X)$ is determinate, not a random variable, when conditioned on $X$
- __keep this in mind!__ the whole purpose of many machine learning models is to model $E(Y\mathbin{\vert}X=x)$ we will see how the models that we talked about before apply this.
- $E(Y\mathbin{\vert}X=x)$ is usually the best that you can do. Finding the mean of $Y$ given $X$. This is because you can only look at $Y$ though what you have: the information in $X$. The information needed to predict $Y$ that is not in $X$ becomes noise to your model.
- kNN approximates $E(Y\mathbin{\vert}X=x)$ with a direct interpretation. The average value of the region around a point, is the value of that point. Remember that $E(Y\mathbin{\vert}X=x)$ you want the average of $Y$ at evaluated at a _point $X$_ rather than a region around the point.
- As the density of the data increases, and sufficient samples are stacked on any given point such that it could approximate the average, kNN becomes more accurate. Unfortunately, we do not have such large amounts of data. And as the number of features increase, data requirements for kNN grows drastically. Though we would have finer control with more features, the degrees of freedom to control have further increased.

- in the act of setting/assuming $f(x) \approx x^T\beta$, we have defined a model for our data. (model based approach)
- Notice how (2.16) is different from (2.6). This is because the value of $f(x)$ is different. Here, $f(x)\approx x^T\beta$ before we saw that $f(x) = X\beta$. (So X here should have column vectors which represents data, N, as opposed to features, p from before.) Also, (2.16) is different in that $X$ represents the whole distribution as it uses $\beta$, rather than just using training set points which $\hat{\beta}$ from (2.6) uses.

- why does additivity matter? Why does this relate to $E(Y\mathbin{\vert}X=x)$? We will see later on, how additivity represents the different components that make $Y$. 
- augumenting kNN with additivity helps resolve the high dimensional problem. Each kNN can done locally for low dimensons, across all of the dimensions, then summed.

- why does $E\mathbin{\vert}Y - f(X)\mathbin{\vert}$ end up being the median($Y$\|$X=x$)? See it [here]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#l1_to_median).
- "robust" in this case means, how much an outlier can affect your model.

- for classification use an L matrix which assigns a loss if a class was missclassified given another.
<!-- code the optimal bayes decision boundary -->

- in the last paragraph, they pose the situation where there were 2 classes (these classes should be: item exists, item doesn't exist) for classification, and used a binary regression Y for this case (Dummy variable Y approach). For k classes we would have $Y_k$, a Y for each class. The reason why we would want to use Y is, now we can apply mean squared error, and minimize that.
- problems might include: the fact that $\hat{f}(x)$ doesn't have to be positive (regression can also be negative). This is a problem if we wanted each $Y_k$ to be a probability, which can't be negative.

## Chapter 2.5: Local Methods in High Dimensions
- _curse of dimensionality_, the amount of data that we need as the dimensions grows, increases exponentially.

- how they got $e_p(r) = r^{1/p}$: $r$ is the volume, the volume of a hypercube is $r = e^p$ where p is the number of dimensions, and e is the edge length. r represents our sample data (or neighborhood for kNN) from the true distribution of the total hypercube (in this case it is the unit hypercube). This means that our data will represent a smaller chunk of the total cube as we increase the dimensionality. The statement that they make for kNN where:
> to capture 1% of the data for a local average, we would need 63% of the range of each input variable.
- Remember, if we decrease the r, (amount of data captured) our variance would increase.

- they also mention that the points will converge to the edges. 

- Question: I'm not sure of an intuitive expanation for this. Why will points converge to the edges? There are many weird things that could happen in high dimensions, where a circle inscribed in a square will end up becoming bigger than that square. Why does this happen?
<!-- program this high dimensionality -->

- extrapolation happens as it is more likely that your new input will fall on the edges. See [this](https://stats.stackexchange.com/questions/206295/curse-of-dimensionality-why-is-it-a-problem-that-most-points-are-near-the-edge) for more.

- problem setup at the bottom of page 23: N =1000 samples along a range of [-1,1] each of the p dimensions. Test our model with at $x_0$ = 0. 

- here is a [proof of (2.25)](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf). $\hat{y}$ seems to be the best estimation of y you can get, based on the information from X.

- Figure 2.7 has a very good visual of how points get further, if you were to look at the top right corner, where the orange would be the kNN for 1D, and the blue would be the kNN for 2 dimensions. 
<!-- program this figure 2.7 and 2.8 -->

- our actual answer is in the shape of a hill that peaks at x=0. So if we were to use kNN, we would usually be less than the actual value, and with the increase in dimensions, the points will be increasingly further away.

- why does some have high variance and some other functions have high bias? If we were to base this off our idea that the data will converge to the edges as the dimensions increase, we see that for figure 2.7, most data points will be at the bottom of the curve, causing for high bias. This bias is due to the fact that answers will tend to be close to 0, even though the actual test value at x = 0 should be 1. There is inherent bias in the data. Here the variance doesn't grow as the points converge to a value of 0. Variance for figure 2.8 will grow, as when the data grow towards the edges, the y values could take on 0, or a very high value. This causes high variance in the data, and as our model predictions are highly dependent on the data (kNN) then we will have high variance in our model predictions.

- the linear model is unbiased (should be a different kind of bias with regards to structural assumptions), as a bias is defined as the difference between the expected value of our prediction and the expected value of the true relationship of X and Y. In our case, the relationship was defined as $Y = X^T\beta + \epsilon$, whereas our linear model is exactly $X^T\beta$. Therefore there is no bias in our linear model.

- [These solutions](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf) include a solid proof for (2.27) and (2.28), but I wanted to further clarify why you would approach some of the steps. My question is, _How would you even begin to solve this?_ since we want to get the biases and the variances the equation, $y_0-\hat{y_0}$, we should seperate them into mean and variance, this is as simple as subtracting the mean from each term that is a non-determinate variable (variables that have a random component). For example, $x_0^T\beta$ is the average which could be subtracted from $y_0$ and $E_\tau\hat{y_0}$ is the average that could be subtracted from $\hat{y_0}$. So here, we can put these terms in the equation, then factor it. The rest of the work is just a bit of manipulation and substituting the values that we used before, as well as leveraging the assumptions. Also keep in mind that you can leverage the trace function whenever the dimension of solution is 1x1.
<!-- program to show XTX = NCov(X) -->

- The reason why the blue line has a lower ratio for figure 2.9, is that linear regression's bias is shown with a cubic function.

## Chapter 2.6: Statistical Models, Supervised Learning and Function Approximation
- how to overcome dimensionality by incorporating leveraging known, special structure within the data.

### Section 2.6.1: A Statistical Model for the Joint Distribution Pr(X,Y)
- $\epsilon$ is assumed to represent all the information about Y that we don't take into account with X. What if this is not the case? What if the variance in Y is not independent of X? 
- Before, we were only finding the mean of Y through $E(Y\|X)$. If the error is known to be dependent on X, then we can leverage this information as well. We can use $Var(Y\|X)$.
- this is usefull for classification which normally aims to model a distribution over each class.

### Section 2.6.2: Supervised Learning
- Learning by example: using the $y$ from your training data to correct your current prediciton.

### Section 2.6.3: Function Approximation
- Function approximation uses a mathematical/statistical approach, rather than biological/human reasoning approach.
- approximators are your model of $Y\|X$: your $f(x)$.
- we saw before that building domain knowledge/assumptions we know about our data could decrease the error, and scale better in higher dimensional spaces. Though it is not limited to this use, we can uses linear basis expansions on our features (applying a linear/nonlinear function on them) as a way to implement domain knowledge. 

- The equation for cross entropy seems off (2.36).
- for an intuitive sense of cross entropy which is measure of similarity between two distributions, usually a modelled distribution and the true distribution, see [here](https://www.youtube.com/watch?v=ErfnhcEV1O8)

## Chapter 2.7: Structured Regression Models
- Structured models allow us to leverage any known structure in the data.

### Section 2.7.1: Difficulty of the Problem
- with a limited training sample set, there could be many different functions for a model that fit across the sample set to a similar degree of accuracy. Restricting the amount of different functions that the model can take will help the model decide on what function to represent. However, we need to choose the restriction carefully, as it should be applicable to our data.
- a type of constraints/restriction that we might impose is the assumption that in small neighborhoods of the data, the behavior/outputs would be similar. This means that in a local area, we can approximately use a constant, linear or a low order polynomial to fit it. The smaller our neighborhoods are, the more complex our model could be.
- rather than specifying the neighborhood, we can give a metric to our model, which can then decide for themselves what a neighborhood should be. We can also have our model adaptively find this. This helps with dimensionality issues, as it is similar to defining a global structure, and will help with dimensionality issues.
- This means that naively defining neighborhoods will cause dimensonality issues. (what if your neighborhood is not large enough to capture the entire area of similar behavior? Then we will need more data to compensate.) (I think that this will be explained more in detail in the future.)

## Chapter 2.8: Classes of Restricted Estimators
- non parametric is a type of model, where the data will supply the structure, rather than using builtin assumptions.

### Section 2.8.1: Roughness Penalty and Bayesian Methods
- we can carry information between close neighvorhoods, so that local regions don't vary too much. cubic smoothing spline is a spline (interpolation between points) that takes into account second derivative information, which signifies how much the function changes it's overall direction. 
- This penalizes how deviant a function is from a straight line.
- the penalty is a prior since it imposes a prior belief about the system. In the spline case, it assumes that there is a degree of smoothness in the optimal structure of the model.

### Section 2.8.2: Kernel Methods and Local Regression
- Methods that fit data locally. 
- Kernels are functions that provide a weight to each of the local points in a neighborhod.
- Nadaraya-Watson weighted average takes into account the surrounding $y_i$, not only the current data point $y_i$.
- $f_{\hat{\theta}}(x_i)$ is the local regression, but our overall goal with this local estimate, is to minimize the global estimate. Just as how we use a sample set (training set) to try to minimize across the whole set.
- remember, that these functions suffer from dimensionality problems.

### Section 2.8.3: Basis Functions and Dictionary Methods
<!-- program basis functions -->
- we can apply a function onto our inputs directly, this will help us if there isn't a directly linear connection between the inputs and outputs. (2.43) is linear even if h(x) is nonlinear, as applying $\theta$ and adding the terms is linear (only doing addtion and multiplication, no sines, squares, etc.). A nonlinear expansion would be if you were to apply a sigmoid after the $\theta$. I think why appling before or after is important, is because we are minimizing with respect to $\theta$, so if a nonlinear function is applied before $\theta$, then the derivative process shouldn't be affected. If we were to apply a function afterwards, then taking the derivative won't be as straight forward.
- we can make predefined (by us) piecewise polynomial splines locally, that are connected together. (these connections points are called knots)
- radial basis pick a certain point, using it as center (called the centroid), then expands outwards. For example, the gaussian kernel from before is like this.
- it would be nice if we have an adaptive knot/centroid placement/best basis type based on the data. Choosing the best basis type is called, dictionary method, which contains a dictionary of different basis function methods and chooses the best one. 
	- An Idea: are not a necessity to understand, and is just there for my own notes, as any new ideas happen.
	- thigh marked with Question: are unanswered questions of mine that are about the content.: It might be accurate if we provide a way of disentanglement for the performance/traits of different basis, used for categorization of different basis for a task. These features could be evaluated based on using a well crafted test dataset to evaluate new basis functions. Maybe, this might lead to the syntesis of new basis functions methods. This will help with accuracy and search speed, as searching for methods by category of their properties would be efficent, being both a direct way to show correlation between the data and the category, as well as being an interpretable/explainability method.

- greedy algorithms are algorithms that are based on the assumption that finding the best, current solution is the best step for the global one. So they will step in the best direction for every step. Though this is not always the case as the previous steps might be correlated/ not independent of each other, where a local best step might not be the optimal step globally. The reason why greedy methods help, is that they allow for tractable calculations to be made. We will see this later on.

## Chapter 2.9: Model Selection and the Bias-Variance Tradeoff
<!-- program this bias variance tradeoff effect -->
- bias variance tradeoff is seems to be the effect of underfitting vs overfitting. Underfitting happens when the model can't model complex data (low model complexity). Overfitting is when the model can model complex data, but this might lead us to memorizing the training set, and thus not being able to learn/represent the underlying principles of the structure of the data. An example of overfitting is 1-nearest-neighbor. 


 