---
layout: default
title: "ESLII Chapter 2 Reference"
---
# Reference for ESLII Chapter 2
Here is the reference for chapter which includes explanations, and definitions for chapter 2 of *The Elements of Statistical Learning*.

### <a id="affine_set"></a>Affine Set
An affine set is a continuous set/space of points, similar to a [convex set](#convex_set), but rather than just including the line segment in between the two point, an affine set includes the points on the line, if both ends of the straight line was extended to infinity. So rather than including the points on just a infinite or finite segment, it includes the points on an infinite line. (convex sets can be infinite as well as finite. Affine sets are only finite.)

Check out [this Quora answer](https://www.quora.com/What-the-relationship-between-affine-set-and-convex-set) or [this stackexchange answer](https://math.stackexchange.com/questions/88750/convexity-and-affineness) for information on affine vs convex sets.

### <a id="convex_set"></a>Convex Set
A convex set is a region of space, where given any 2 points in that region/set, if we were to draw a straight line segment connecting the two points, every point on that line should be in the convex set as well. This means that regions that look line a "U" are not convex since you can pick out two points (eg. on the tail ends of the U) where the line inbetween is not included in the set. 

For more information watch [this youtube video](https://www.youtube.com/watch?v=VcTIOQpRG1o)!

### <a id="l1_to_median"></a>How does $E\mathbin{\vert}Y - f(X)\mathbin{\vert}$ end up being the median($Y$\|$X=x$)?
Lets look at this closely. $Y - f(X)$ is the error of your model. 

_If this error was squared (L2 error)_: then larger errors for a point x_l, will be weighted more than smaller errors by a given point x_s. A small shift in $f(x)$ towards the larger error point will reduce the error by much more than a shift of the same amount towards that small error point. This means that the model will be biased towards solving the larger errors first. Which might be what you want, but this also means that outliers could be a huge problem. Also, if you wanted to build a model high in variance, this loss might cause constrain that (Eg. Generative models for art. You wouldn't want the same art piece to be generated each time.)

However _if we used the abolute of this error (L1 error)_: then shifts towards larger values will decrease the error by the same amount as shifts towards the smaller values. This will cause us to find some sort of equilibrium at the median of the datapoints, where if you shifted your prediction up, the error of the points below would increase, while the points above would decrease, and vice versa for a shift downwards. This will be much more helpful with regards to outliers.

I highly suggest drawing out a numberline with a few points and finding the minimum absolute error to prove that it will be the median.

check out [this stack exchage answer](https://stats.stackexchange.com/questions/34613/l1-regression-estimates-median-whereas-l2-regression-estimates-mean) for more information.

### <a id="structural_assumptions"></a>Structural Assumptions
Structural assumptions assumptions made about the global relationship between all x and y. For Example, linear relationship.

We make these assumptions to decrease the amount of data, as you won't need to use the data to determine what the structure is. 

Another reason we make these assumptions, is that our prediction could be more accurate if our assumptions are correct. We will see this later in Chapter 2.5 (Figure 2.7, 2.8)


### <a id="voronoi_tessellation"></a>Voronoi Tessellation/Voronoi Diagrams
There are the splitting of space into different regions, based on distance to the closest point. This is similar to a decision boundary for each point in kNN. [Seeing a picture](https://en.wikipedia.org/wiki/Voronoi_diagram) would help.


### <a id="minimizing_pointwise"></a>Minimizing _pointwise_
Minimizing pointwise means minimizing, given the samples that we have, as opposed to the whole $X$ distribution (which we don't have). So basically, just minimizing over our training set. The effect that this would have on the equations is $X$ becomes $X=x$ where we give it the training set of x.

### <a id="robust"></a>Robustness
For a model to be "robust" in this case means, how much an outlier can affect your model. We would want to know this as if a single data point can heavily influence our model, we would either need to introduce more datapoints, or do more work to ensure the data is cleaned.

### <a id="curse_of_dimensionality"></a>Curse Of Dimensionality
The amount of data that we need increases exponentially as the amount of dimensions in the input. As the dimensionality increases, the amount of space that the model has to adjust to increases. Why would we want to increase the dimensionality? Remember that we can see the dimensionality as being more features that we can define our output with, thus, if our training set sufficiently represents the true distribution, we could possibly have features that would decrease the error/uncertainty in our output. (Possibly, since the features might be orthogonal to the output.)