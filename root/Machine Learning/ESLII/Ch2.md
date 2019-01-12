---
layout: default
title: Ch2
permalink: /ml/ESLII/Ch2/
---
# ESLII Chapter 2: Overview of Supervised Learning
<!-- Introduction/overview of chapter -->
## General Tips for Chapter 2:
### Tip #1: Understanding Equations
When looking at equations, it would help to think simple when analyzing equations for an intuitive sense of whats going on. I like to think of N as being equal to 1 unless otherwise stated (or if there is a reason it can't be), where you only have one data point. Also, know what assumptions you can make, and how that translates to math, when trying to solve the equations yourself. There are many subtlties in the wording and equations. For example, if you have $XX^T$ vs $X^TX$, _why do we use one or the other?_ Assuming that X is zero centered, and is a vector, then these would both represent variance. $XX^T$ would represent a matrix of how much each of the features covary (covarince matrix scaled by N). If you took the diagonal of this matrix as a vector, then it would represent a elementwise multiplication of the $X$ vector with itself. $X^TX$ on the other hand, represents sum of the total variance across each feature. In otherwords, it is the sum of the diagonal of the $XX^T$ matrix. You can see how, each of these differs in a practical sense. One gives a detailed breakdown of the variances across each feature, and the other gives an overall view of the variance in the system. 

Make sure that you know the purpose behind each equation, and each operation. Looking at equations as just numbers, variables and functions is like reading a book by just paying attention to each single word, instead of taking in what the overall story is. Instead, keep in mind what each aspect of an equation represents, and what it means to be doing a certain operation. In the future, this will help you translate theory into math, then further on into effcient algorithms. This doesn't mean to neglect proving them yourself, proving them yourself will get you familiar with the tools you can use.

One thing that will help with representations, is by looking at an equation vector-wise. In a $Nxp$ matrix $X$, $N$ signifies the datapoints, $p$ signifies the features. Becareful of which part is multiplying which, sometimes, one matters more than the other. 

### Tip #2: Other General Tips:
- Bookmark pg 12 for it's equations, I found it helpful to look back on this in future chapters. (You don't want to be deriving it every single time, though you should derive it at least once.)
- things marked with _An Idea:_ are not a necessity to understand, and is just there for my own notes.
- things marked with _Question:_ are unanswered questions of mine that are about the content.
- [__Code is available! Click here!__]({{ site.github.owner_url }}/yukunchen113.github.io/tree/master/root/Machine%20Learning/ESLII/Ch2_code) This code is for the various parts in the chapter, and each files will be specified below.


## Chapter 2.3:

### Introduction

what are [structural assumptions]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#structural_assumptions)?

linear models assume linearity in the structure of the data, so they might be inaccurate if the data is non linear.

kNN do not make any strong assumptions about the structure of the data, instead, it uses the data to create the structure. What does "strong" mean? While linearity assumes that the mapping from X to Y is linear, which is a global assumption. kNN is more of a local assumption that relies heavily on the training data in a local area. But because of this heavy reliance on the training data, with the addition of a new data point, the model can change significantly. This is bad as we would have to select our training data very carefully, a slight difference could give us a very different outcome.

I think that structure is assumed when you have a model, and your data is used to tune the finite number of parameters in the model. Linear models seem to do this, and kNN don't. It's similar using data to fill in the blanks in a partly constructed paragraph vs using data to write a new paragraph. You will need more data if you don't make structural assumptions, as well as there would be high variance.

### Section 2.3.1: Linear Models and Least Squares
what is an [affine set]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#affine_set)?

Here are some points about the (p+1) dimensional input-output space:

The set ($X, \hat{Y}$) is represented in a p dimensional space, where each dimension is represents a feature. Remember that our estimated value, $\hat{Y}$ is just a linear combination of the features in $X$, so therefore it has to exist within the space that $X$ represents.

The last dimension of the (p+1) dimensions (the +1 dimension, the other p dimensions are from ($X,\hat{Y}$)) represents the combination of all the uncertain/unknown features in $Y$ (the actual output). This component is not represented by the hyperplane/features of $X$. For example, when we include the $\hat{\beta_0}$ as a component of $X$, $\hat{\beta_0}$ is not part of the uncertainty in $Y$ anymore, and becomes a component of $X$, though there are still other factors that contribute to that uncertainty. I also want to point out that even though the features are not necessarily orthogonal to each other, they are still used as the basis of the space, therefore the input-output space doesn't have to be orthogonal.

The small $x_i$ in (2.3) represents a data vector of size p.

[This site talking about transpose properties](https://math.vanderbilt.edu/sapirmv/msapir/prtranspose.html) is good for a quick reference for solving proofs. Though you should definitely work them out your self to see if they make sense. (Just quickly draw a matrix and prove it to yourself)

For information on matrix differentiation see [this great site](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf), _section 5. Matrix Differentiation_ for identities and their proofs (2.4) -> (2.5). If you go through in detain, it gives a very good idea on matrix differentiation. Again, you should also write out the proofs yourself whenever you are slightly unsure (or even when you are sure!).


What does a singular $X^TX$ represent? What does it tell us about our data? $X^TX$ represents a matrix whose elements are dot products between each feature as a p-dimensional vector. Multiplication happens between features within datapoints (feature x in datapoint i multiplies with feature y in datapoint i). This is an uncentered covariance between features, where $X^TX$ represents covariance, but since the features don't have a mean of 0, it is uncentered. For a row in $X^TX$ to be 0, (causing a singular matrix) a feature must be orthogonal (dot product = 0) to every feature (including itself!), so this means that the matrix will be singular if all the elements in a feature vector is 0. So across all data points, a feature is always 0. (Though I didn't exhaustively prove this, so don't quote me on this one!)

If a mixture of gaussians is not tightly clustered, we can use a random variable to determine which component gaussian in the mixture to sample from, then, sample from that gaussian. 

Code for Scenario 1 and Scenario 2, as well as gaussian mixtures is available. Implemented with Python and Numpy. Called _gaussian_mixture.py_


### Section 2.3.2: Nearest Neighbor Methods
what are [_voronoi tessellation_ or _voronoi diagrams_]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#voronoi_tessellation)?

Becareful of overfitting (k=1), use a test set to determine if your model holds up.

The _effective_ number of parameter for kNN is $N/k$, which is a rough measure that is proportional to the max number of regions (would be the number of neighborhoods, if the neighborhoods didn't overlap)

kNN is good for scenario 2 as scenario 2 has many small regions. As kNN is inherently regional, it will perform well with scenario 2, given a proper k value, though it will have very noisy boundaries. 

### Section 2.3.3: From Least Squares to Nearest Neighbors
Code is avaliable for _"exposing the oracle!"_ on pg. 16. Code is available in the Chapter 2 repository above, called, _oracle.py_

Basis expansion is a transformation on the inputs, rather than just taking the inputs at face value, and expecting a linear relationship with the output. Eg. using $X^2$ rather than just $X$.

Projection pursuit defines a set of methods projects your high dimensional data onto a lower dimension, trying to keep the most informative/interesting parts. Eg. PCA. 

## Chapter 2.4: Statistical Decision Theory
what do you mean by [_minimizing pointwise_]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#minimizing_pointwise)?

we can replace $f(X)$ with $c$ since $f(X)$ is determinate, not a random variable, when conditioned on $X$. 

__Keep this in mind! the whole purpose of many machine learning models is to model $E(Y\mathbin{\vert}X=x)$__ we will see how the models that we talked about before apply this. 

$E(Y\mathbin{\vert}X=x)$ is usually the best that you can do. Finding the average of a point in $Y$ given a point in $X$ for all $Y$ and $X$. This is because you can only predict at $Y$ with what you have: the information in $X$. The information about $Y$ that is not in $X$ becomes noise to your model.

kNN approximates $E(Y\mathbin{\vert}X=x)$ with a direct interpretation. It takes the average value of the region around a point and assumess that approximates the value of at the point. Remember that $E(Y\mathbin{\vert}X=x)$ you want the average at a point $Y$ at evaluated at a _point $X$_ rather than a region around the point (which is what kNN does). 

As the density of the data increases, and sufficient samples are stacked on any given point such that it could approximate the average well enough, kNN becomes more accurate. Unfortunately, we do not have such large amounts of data. And as the number of features increase, data requirements for kNN grows drastically. Though we would have finer control with more features, each feature, which can be seen as degrees of freedom, have further increased.

By setting/assuming $f(x) \approx x^T\beta$, we have defined a model for our data. (model based approach). This is a global structure that we assume will be applicable across all of Y. By constrast, kNN only assumes a local structure, thus allowing the global structure to be controled by the data. I think that it's similar to a heavy plate armour vs a piece of chainmail. The heavy plate linear regression seems to be defined globally as it's overall structure is a predefined shape (of a person in the case of heavy armour), though it could adjust due to the parameters (can move arm and legs). The chainmail piece has locally defined structure, (each small loop of the chain) though the overall shape is still flexible, and would require a person wearing it to define the overall shape.

Notice how (2.16) is different from (2.6). This is because the value of $f(x)$ is different. Here, $f(x)\approx x^T\beta$ before we saw that $f(x) = X\beta$. (So X here should have column vectors which represents data N, as opposed to features, p from before.) Also, (2.16) is different in that $X$ represents the whole distribution as it uses $\beta$, rather than just using training set points which $\hat{\beta}$ from (2.6) uses. This means that they're trying to solve it generally across the wold distribution of $X$

Why does additivity matter? How does this relate/implement to $E(Y\mathbin{\vert}X=x)$? We will see later on, in chapter 3, how additivity represents the different components that make $Y$. 

Augumenting kNN with additivity helps to resolve the high dimensional problem. Also we can have each kNN can be done locally for low dimensons, across all of the dimensions, then summed.

why does $E\mathbin{\vert}Y - f(X)\mathbin{\vert}$ end up being the median($Y$\|$X=x$)? See it [here]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#l1_to_median).

what does it mean for a model to be [robust]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#robust)?

For classification use a loss matrix, which assigns a loss if a class was missclassified given another.

Code and explanation for the bayes decision boundary is in the chapter 2 repository, which is linked above. It plots both mixture distributions for the oracle, and also plots the decision boundary. The file is called _bayes_decision_boundary.py_

In the last paragraph of the section, they pose the situation where there were 2 classes (these classes should be: item exists, item doesn't exist) for classification, and used a binary regression Y for this case (Dummy variable Y approach). For k classes we would have $Y_k$, a Y for each class. Notice how, we are now using a $Y$ for classification. So now we are doing a regression across each class, where the regression predicts the probability of being that class. Why would we do this? The reason why we would want to use Y is, now we can apply mean squared error to minimize Y, which would make solving our problem easier. There might be some problems with using regression for modeling a probability. problems might include: the fact that $\hat{f}(x)$ doesn't have to be positive (regression can also be negative). This is a problem if we wanted each $Y_k$ to be a probability, which can't be negative.

## Chapter 2.5: Local Methods in High Dimensions
what is the [_curse of dimensionality_]({{ site.baseurl }}{% link _posts/2018-12-23-ESLII-Chapter-2-Reference.md %}#curse_of_dimensionality)?

How they got $e_p(r) = r^{1/p}$: $r$ is the volume, the volume of a hypercube is $r = e^p$ where p is the number of dimensions, and e is the edge length of the sample. r represents our sample data (or we can might also see it as a neighborhood for kNN) from the true distribution of the total hypercube (in this case it is the unit hypercube). This equation means that our data will represent a smaller chunk of the total cube as we increase the dimensionality (What happens to edge, as we represent the same r in higher dimensions? Keep r constant, increase p, see what happens to e). The example they use to show how data hungry higher dimensions are for kNN:
> to capture 1% of the data for a local average, we would need 63% of the range of each input variable.

Remember, if we decrease the r, (amount of data captured) our variance (error term, $\epsilon$) would increase.

They also mention that the data points will converge to the edges as we further increase the dimensionality. 

__Question__: I'm not sure of an intuitive expanation for either the increase in volume for dimensionality, or the fact that data goes to the edges. Why does this happen?


<dl>
	<dt>__Answer__:</dt>
	<dd>

	A basis for an intuition on the problem and how to solve it, is 

	<a href="https://www.youtube.com/watch?v=mceaM2_zQd8">

	presented by Numberphile</a>. I will be piggybacking off the problem setup and give my own take on the solution.<br><br>

	Rather than the ball becoming "spikey" like he says at the end, I picture that the bounding spheres are becoming farther away from each other as dimensionality increases when projecting the bounding spheres back to a lower dimension. Here's what I mean; imagine the 3D case, with 8 bounding spheres, and the 2D case with 4 bounding circles. The center sphere in the 3D case is not the same as the one in the 2D case, even if you were to expand the 2D case to 3 dimensions. They have a different centerpoint, and they touch the bounding sphere in a different way. Even though the 4 bounding circles in the 2D case "bound" the center sphere by making contact, in the 3D case, they do not define the <em>range/diameter</em> of the center sphere anymore. Defining diameter is what I call bounding. Again, imagine going from the 2D case to the 3D case. The you can see that the center sphere is not stuck in the same place. The bounding spheres that _do_ define the diameter of the sphere are now spheres on a slanted diagonal cross section of the cube. If we want the the same kind of bounding (bounding = diameter definition) for the 3D case to be transfered back down to the 2D case, we need to cut the 3D bounding spheres along a diagonal, slanted cross section of the cube. (see code for a visual). <br><br>

	If we were to draw this new cross section on paper, it would seem that the bounding spheres now have a gap in between them, and the spheres are no longer touching. This is where the extra bit of diameter comes in. The small gap also allows parts of the center sphere to be close to the edges of the cube. We can imagine if this were to continue in higher dimensions (cutting diagonals, and gaps becoming larger/more abundant), We can see how the center sphere will grow to have a radius bigger than the box due to the increase in gaps.<br><br>

	Connecting this back to our dimensional problem, if we were to imagine the bounding spheres as data points, with the center sphere as the test point, we can see how the datapoints drift become comparitively farther as we increase the dimensionality.<br><br>

	There is code is available for visualizing this. it is called <em>higher_dimensional_spheres.py</em>.

</dd>

</dl>

 


Extrapolation happens as it is more likely that your new input will fall on the edges. See [this](https://stats.stackexchange.com/questions/206295/curse-of-dimensionality-why-is-it-a-problem-that-most-points-are-near-the-edge) for more. This causes your model to perform worse, usually models are bad at extrapolation.

Problem setup at the bottom of page 23: N = 1000 samples along a range of [-1,1] each of the p dimensions. Test our model with at $x_0$ = 0. 

Here is a [proof of (2.25)](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf). $\hat{y}$ seems to be the best estimation of y you can get, based on the information from X.

Figure 2.7 has a very good visual of how points get further, if you were to look at the top right corner, where the orange would be the kNN for 1D, and the blue would be the kNN for 2 dimensions. This gives a clear look on how adding more feaures would cause the distance to our test point to increase as dimensionality does. It's a simple intuitive way.

Our actual answer is in the shape of a hill that peaks at x=0 for figure 2.7. So if we were to use kNN, we would usually get a prediction that is less than the actual value (remember that we are evaluating at x = 0), and with the increase in dimensions, the points will get increasingly further away, thus increasing this gap between the actual and predicted values.

Why does some models have high variance and some other functions have high bias? If we were to think about the idea that the data will converge to the edges as the dimensions increase, we see that for figure 2.7, most data points will be at the bottom of the curve, causing for high bias (predicted is always less than actual, however, the data points themselves are quite close to each other, and thus shows little variance). Our predicted values for this high bias case will tend to be close to 0, even though the actual test value at x = 0 should be 1. We see that here, there is inherent bias in the data, the variance doesn't grow as the points converge to a value of 0. However, variance for figure 2.8 will grow, as when the data for figure 2.8 grows towards the edges, the y values would take on 0, or a very high value. This causes high variance in the data, and as our model predictions are highly dependent on the data (assuming that we are using kNN) then we will have high variance in our model predictions, but low bias, as the expected values of the predicted and actual would be similar to each other.

The linear model is unbiased (This is a different kind of bias than the one with regards to structural assumptions), as a bias is defined as the difference between the expected value of our prediction and the expected value of the true relationship of X and Y. In our case, the relationship was defined as $Y = X^T\beta + \epsilon$, whereas our linear model is exactly $X^T\beta$. Therefore there is no bias in our linear model. Practically, this means that if we have domain knowledge, where we know that our assumptions were proven, we should incorporate that domain knowledge.

[These solutions](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf) include a solid proof for (2.27) and (2.28), but I wanted to further clarify why you would approach some of the steps. My question is, _How would you even begin to solve this?_ since we want to get the biases and the variances the equation, $y_0-\hat{y_0}$, we should seperate them into mean and variance, this is as simple as subtracting the mean from each term that is a non-determinate variable (variables that have a random component). For example, $x_0^T\beta$ is the average which could be subtracted from $y_0$ and $E_\tau\hat{y_0}$ is the average that could be subtracted from $\hat{y_0}$. So here, we can put these terms in the equation, then factor it. The rest of the work is just a bit of manipulation and substituting the values that we used before, as well as leveraging the assumptions. Also keep in mind that you can leverage the trace function whenever the dimension of solution is 1x1.

The reason why the blue line has a lower ratio for figure 2.9, is the model for the blue line is a cubic, and the actual equation is cubic.

## Chapter 2.6: Statistical Models, Supervised Learning and Function Approximation
How to overcome dimensionality by incorporating/leveraging known, special structure within the data.

### Section 2.6.1: A Statistical Model for the Joint Distribution Pr(X,Y)
$\epsilon$ is assumed to represent all the information about Y that we don't take into account with X. What if this is not the case? What if the variance in Y is not independent of the data in X? That the information X contains can satisfy Y's uncertainty? Before, we were only finding the mean of Y using $E(Y\|X)$. If the error is also known to be dependent on X, then we can leverage this information as well. We can use $Var(Y\|X)$ to represent the error. Now that we have variance, this is useful for classification, which normally aims to model a distribution over each class.

### Section 2.6.2: Supervised Learning
Learning passively by example/with a teacher: using the $y$ from your training data to correct your current prediciton. Learing passively because we are feeding in the data and it doesn't have control over what data will be fed in next.

### Section 2.6.3: Function Approximation
Function approximation uses a mathematical/statistical approach on machine learning, rather than biological/human reasoning approach like neural networks.

Approximators are your model of $Y\|X$: your $f(x)$.

We saw before that building domain knowledge/assumptions we know about our data could decrease the error, and scale better in higher dimensional spaces. We can use linear basis expansions on our features (applying a linear/nonlinear function on them) as a way to implement domain knowledge. 

The equation for cross entropy seems off (2.36), it is different from the one shown in the video below. Why can we use either one?
For an intuitive sense of cross entropy, which is measure of similarity between two distributions, (usually a modelled distribution and the true distribution), see [here](https://www.youtube.com/watch?v=ErfnhcEV1O8)

## Chapter 2.7: Structured Regression Models
Structured models allow us to leverage known structure in the data.

### Section 2.7.1: Difficulty of the Problem
With a limited training set, there could be many different functions that could fit across it, with to a similar degree of accuracy on the set. Restricting the amount of different functions that the model can be, will help the model decide on what function to represent. However, we need to choose these restrictions carefully, as they should be representative of our data.

A type of constraint/restriction that we might impose is the assumption that small neighborhoods of the data would behave simiarly, producing similar outputs. This means that in a local area, we can approximately use a constant linear or a low order polynomial to fit it. The smaller our neighborhoods are, the more complex representations our model could be.

Also, rather than specifying exactly the neighborhood, we can give a metric to our model, letting the models decide for themselves how big a neighborhood should be. Our model can adaptively find this. This will help with dimensionality issues, as it is similar to defining a more global structure to the model.

This means that naively defining small, neighborhoods will cause dimensonality issues. For example, what if your neighborhood is not large enough to capture the an area of similar behavior? We will be uselessly representing a common area with more neighborhoods than necessary. We will need more data to compensate. I think that this will be explained more in detail in the future.

## Chapter 2.8: Classes of Restricted Estimators
Non parametric is a type of model where the data will supply the structure/define the model. This means that there are an infinite number of parameters to model. 

### Section 2.8.1: Roughness Penalty and Bayesian Methods
We can carry information across close neighborhoods, so that local regions don't vary too much. An example technique that we can apply is the cubic smoothing spline, which is a spline (interpolation between points) that takes into account second derivative information. Second derivative shows how much the function changes it's overall direction, if we add this to the loss function we will penalize how deviant the function is from a straight line.

The penalty imposes a prior belief about the system which we have onto the system. In the spline case, it assumes that there is a degree of smoothness in the optimal structure of the model.

### Section 2.8.2: Kernel Methods and Local Regression
Kernel metheds are methods that fit data locally. Kernels are functions that provide a weight to each of the local points in a neighborhood. They focus on a point as the center, and apply a function to the points around the center point, using the distance to the center point. For example, the Nadaraya-Watson weighted average takes into account the surrounding points $y_i$, not only the current data point $y_0$, the effect of the $y_i$ decrease as you get further away from the center point.

$f_{\hat{\theta}}(x_i)$ is the local regression. Remember that our overall goal with minimizing this local estimate, is to minimize the global estimate. Just as how we use a sample set (training set) to try to minimize across the whole set, we assume that minimizing the local structure will also be minimizing the global structure.

However, remember that these local functions suffer from dimensionality problems.

### Section 2.8.3: Basis Functions and Dictionary Methods
We can apply a function on our inputs directly and feed that into the model, this will help us if the output doesn't have a directly linear connection with the inputs. Notice how (2.43) is defined as linear even if h(x) can be nonlinear. A linear expansion is defined as such when the $\theta$'s are linear. A nonlinear expansion would be if you were to apply a sigmoid after the $\theta$s. What's the difference between appling the nonlinearity before or after? Remember that we are minimizing with respect to $\theta$, so if a nonlinear function is applied before applying $\theta$ (2.43), then the derivative process shouldn't be affected as we are deriving with respect to $\theta$. Otherwise, when applying the non linearity afterwards, taking the derivative won't be as straight forward.

We can make predefined piecewise polynomial splines in local areas and connect them together. The connections points between two splines are called knots

Radial basis pick a certain point, using it as center (called the centroid), then expands outwards. For example, the gaussian kernel from before (Nadaraya-Watson) is like this.

It would be nice if we have an adaptive knot/centroid placement/best basis type based on the data. Choosing the best basis type for each neighborhood is called a dictionary method. Where you would have a dictionary of different basis function methods, and the model chooses the best one. 

An Idea: It might be accurate if we provide a way of disentanglement for the performance/traits of different basis, used for categorization of different basis for a task. These features could be evaluated based on using a well crafted test dataset to evaluate new basis functions. Maybe, this might lead to the syntesis of new basis functions methods. This will help with accuracy and search speed, as searching for methods by category of their properties would be efficent, being both a direct way to show correlation between the data and the category, as well as being an interpretable/explainability method.

Greedy algorithms are algorithms based on the assumption that finding the current best solution is also the best step for the global one. So they will step in the best direction for every step. Though this is not always the case as the previous steps might be correlated/ not independent of each other, where a local best step might not be the optimal step globally. The reason why greedy methods help, is that they allow for tractable calculations to be made. We will see this later on.

## Chapter 2.9: Model Selection and the Bias-Variance Tradeoff
Bias variance tradeoff is seems to be the effect of underfitting vs overfitting. Underfitting happens when the model can't model complex data (low model complexity). Overfitting is when the model can model complex data, but this might lead us to memorizing the training set, and thus not being able to learn/represent the underlying principles of the structure of the data. An example of overfitting is 1-nearest-neighbor. 


 