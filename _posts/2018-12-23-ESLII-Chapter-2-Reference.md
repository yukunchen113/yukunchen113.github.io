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