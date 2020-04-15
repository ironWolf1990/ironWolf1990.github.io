---
title: Training and Optimization
layout: post
category: Machine Learning
date: 2020-04-12
excerpt: "Gradient Descent iteratively adjusts the values, using calculus, so that they minimize the given cost-function"
abstract: "Gradient Descent is a method used while training a machine learning model. It is an optimization algorithm, based on a convex function, that tweaks it’s parameters iteratively to minimize a given function to its local minimum."
---

Optimization of the training criterion of a multi-layer neural network is difficult because there are numerous local minima. It can be demonstrated that finding the optimal weights is *NP-hard*. Find a good local minimum, or even just a sufficiently low value of the training criterion. We are interested in generalization error and not just the training error (the difference between *close to a minimum* and *at a minimum* is often of no importance). There is no analytic solution to the minimization problem, we are forced to perform the optimization in an iterative manner.

Two fundamental issues guide the various strategies employed in training MLPs:<br/>
+ training as efficiently as possible, i.e., getting training error down as quickly as possible, avoiding to get stuck in narrow valleys or even local minima of the cost function,
+ controlling capacity so as to achieve the largest capacity avoids overfitting, i.e., to minimize generalization error.

Given we have a fixed training set $${(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})}$$ of $$m$$ training examples. We can train our neural network using batch gradient descent. In detail, for a single training example $$(x,y)$$, we define the cost function with respect to that single example to be :

 $$\begin{align}
 J(W,b; x,y) = \frac{1}{2} \left\| h_{W,b}(x) - y \right\|^2.
 \end{align}$$

This is a (one-half) squared-error cost function. Given a training set of $$m$$ examples, we then define the overall cost function to be:

 $$\begin{align}
 J(W,b) &= \left[ \frac{1}{m} \sum_{i=1}^m J(W,b;x^{(i)},y^{(i)}) \right] + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
 \end{align}$$

*first term in the definition of $$\operatorname{J}(W,b)$$ is an average sum-of-squares error term. The second term is a regularization term (also called a weight decay term) that tends to decrease the magnitude of the weights, and helps prevent overfitting.*

### Weight Decay

Learning rate is a parameter that determines how much an updating step influences the current value of the weights. While weight decay is an additional term in the weight update rule that causes the weights to exponentially decay to zero, if no other update is scheduled.Gradient descent tells us to modify the weights
$$w$$ in the direction of steepest descent in $$J$$(cost function) :

 $$\begin{align}
 \Delta w &= − \eta\text{ }\nabla \operatorname{J}(w) \\
 \Delta w_i &= - \eta \frac{\partial J}{\partial w_i}
 \end{align}$$

+ where $$\eta$$ is the learning rate, and if it's large you will have a correspondingly large modification of the weights $$w_i$$(shouldn't be too large, otherwise you'll overshoot the local minimum in your cost function).
+ to effectively limit the number of free parameters in your model so as to avoid over-fitting, it is possible to regularize the cost function. An easy way to do that is by introducing a *zero mean Gaussian prior* over the weights, which is equivalent to changing the cost function to $$\widetilde{J}(\mathbf{w})=J(\mathbf{w})+\frac{\lambda}{2}\mathbf{w}^2$$. In practice this penalizes large weights and effectively limits the freedom in your model. The regularization parameter $$\lambda$$ determines how you trade off the original cost with the large weights penalization.
+ applying gradient descent to this new cost function we obtain:

 $$\begin{equation}
 w_i \leftarrow w_i-\eta\frac{\partial J}{\partial w_i}-\eta\lambda w_i.
 \end{equation}$$

+ The new term $$-\eta\lambda w_i$$ coming from the regularization causes the weight to decay in proportion to its size. Usually weight decay is not applied to the bias terms $$b^{(l)}_i$$. Applying weight decay to the bias units usually makes only a small difference to the final network.

It is not surprising that weight decay will hurt performance of your neural network at some point. Let the prediction loss of your net be $$L$$ and the weight decay loss $$R$$. Given a coefficient $$\lambda$$ that establishes a tradeoff between the two, one optimises

 $$J=L + \lambda R$$

At the optimium of this loss, the gradients of both terms will have to sum up to zero:

 $$\nabla L = − \lambda \nabla R$$

This makes clear that we will not be at an optimum of the training loss. Even more so, the higher $$\lambda$$ the steeper the gradient of $$L$$, which in the case of convex loss functions implies a higher distance from the optimum.

### Symmetry Breaking

During forward propagation each unit in hidden layer gets signal: $$a_i=\sum_{i}^{N}(W_{ij}x_i)$$. If all weights are initialized to $$1$$, each unit gets signal equal to sum of inputs (and outputs `sigmoid(sum(inputs))`). If all weights are $$0$$, which is even worse, every hidden unit will get zero signal.

+ it is important to initialize the parameters randomly, rather than to all $$0$$s. If all the parameters start off at identical values, then all the hidden layer units will end up learning the same function of the input (more formally, $$W^{(1)}_{ij}$$ will be the same for all values of $$i$$, so that $$a^{(2)}_1 = a^{(2)}_2 = a^{(2)}_3 = \ldots$$ for any input $$x$$)
+ error is propagated back through the weights in proportion to the values of the weights. This means that all hidden units connected directly to the output units will get identical error signals, and, since the weight changes depend on the error signals, the weights from those units to the output units must always be the same. The system is starting out at a kind of unstable equilibrium point that keeps the weights equal, but it is higher than some neighboring points on the error surface, and once it moves away to one of these points, it will never return.

 To train our neural network, we will initialize each parameter $$W^{(l)}_{ij}$$ and each $$b^{(l)}_{i}$$ to a small random value near zero (e.g. normal distribution $$(0,ϵ^2)$$ for some small ϵ ~ 0.01), and then apply an optimization algorithm. The random initialization serves the purpose of *symmetry breaking*.

## Most Common Techniques

1. *early stopping:* this is one of the most popular and most efficient techniques(does not work well when the number of examples is very small). We use a *validation set* of examples held out from training via gradient descent to estimate the generalization error as the iterative learning proceeds (normally, after each epoch, measure the error on the validation set). We keep the parameters corresponding to the minimum of this estimated generalization error curve (stop when this error begins to climb or if a better minimum is not found in a certain number of epochs).
2. *controlling the number of hidden units:* This number directly influences the capacity. In this case we must unfortunately perform several experiments, unless using a constructive learning algorithm which adds resources if and when they are needed). We can use a validation set or cross-validation to estimate the generalization error. This estimate is noisy (if there are few examples in the validation set). When there are many hidden layers, choosing the same number of hidden units per layer seems to work well. By contrast, when the number of hidden units is too small, the effect on the generalization error and the training error can be much larger. Generally choose the size of networks empirically.
3. *weight decay:* is a regularization method (for controlling capacity, to prevent overfitting). The aim is to penalize weights with large magnitude. We add the penalty $$\lambda \sum_i \theta_i^2$$ to the cost function. This is known as L2 regularization, since it minimizes the L2 norm of the parameters. Sometimes this is applied only to the weights and not to biases.<br/>We must run several leaning experiments and choose the penalty parameter $$\lambda$$ that minimizes the estimated generalization error. This can be estimated with a validation set or with cross-validation.<br/>
A form of regularization that is more and more used as an alternative to L2 regularization is L1 regularization, which has the advantage that small parameters will be shrunk to exactly 0, giving rise to a sparse vector of parameters. It thus minimizes the sum of the absolute values of the parameters (the L1 norm).
<figure style="width: 650px" class="align-center">
  <img src="/images/deep-learning/reg_strength.jpeg" alt="Regularization Strength">
  <figcaption>Changing the regularization strength makes its final decision regions smoother with a higher regularization.<br/>ConvNetJS Demo @ stanford.edu </figcaption>
</figure>
<figure style="width: 650px" class="align-center">
  <img src="/images/deep-learning/layer_size.jpeg" alt="Layer Size">
  <figcaption>Overfitting occurs when a model with high capacity fits the noise in the data instead of the (assumed) underlying relationship.<br/>ConvNetJS Demo @ stanford.edu .</figcaption>
</figure>

## Output Non-Linearities for Training

Let $$f(x)=r(g(x))$$ with $$r$$ representing the output non-linearity function. In supervised learning, the output $$f(x)$$ can be compared with a target value $$y$$ through a loss functional $$L(f,(x,y))$$. Common loss functionals, with the associated output non-linearity:

+ *ordinary (L2) regression:* no non-linearity $$(r(a)=a)$$, squared loss<br/> $$L(f,(x,y)) = \left\| f(x) - y \right\|^2 = \sum_i^{} (f_i(x) - y_i)^2$$
+ *median (L1) regression:* no non-linearity $$(r(a)=a)$$, absolute value loss<br/> $$L(f,(x,y)) =$$ l $$ f(x)-y $$ l $$ = \sum_i^{} $$ l $$f_i(x) - y_i$$ l
+ *2-way probabilistic classification:* sigmoid non-linearity ($$r(a)=sigmoid(a)=\frac{1}{(1+e^{-a})}$$) applied element by element, cross-entropy loss:<br/> $$L(f,(x,y))= -y \log f(x) -(1-y)\log(1-f(x))$$ for $$y$$ binary.<br/>*Note: the sigmoid output $$f(x)$$ is in the interval $$(0,1)$$, and corresponds to an estimator of $$P(y=1$$* l *$$x)$$. The predicted class is $$1$$ if $$f(x)>\frac{1}{2}$$*
+ *multiple binary probabilistic classification:* each output element is treated as above
+ *2-way hard classification with hinge loss:* no non-linearity $$(r(a)=a)$$ and the hinge loss:<br/>$$L(f,(x,y))= \max(0,1 - (2y-1) f(x))$$, for binary y.<br/>*Note: is the SVM classifier loss.*
+ the above can be generalized to *multiple classes* by separately considering the binary classifications of each class against the others.
+ *multi-way probabilistic classification:* softmax non-linearity $$(r_i(a) = e^{a_i}/\sum_j e^{a_j}$$), one output per class with the negative log-likelihood loss:<br/>$$L(f,(x,y)) = - \log f_y(x)$$<br/>*Note: $$\sum_i f_i(x)=1$$ and $$0<f_i(x)<1$$*

## Architecture

+ in theory one hidden layer suffices, but this theorem does not say that this representation of the function will be efficient (with the exception of convolutional neural networks). Sometimes one obtains much better results with two hidden layers or obtain a much better generalization error with even more layers. But a random initialization does not work well with more than two hidden layers.
+ for regression or with real-valued targets that are not bounded, generally better to use linear neurons in the output layer. For classification, generally better to use neurons with a non-linearity (sigmoid or softmax) in the output layer.
+ for certain cases, direct connections between the input and the output layer can be useful. In the case of regression, they can also be directly initialized by a linear regression of the outputs on the inputs. The hidden layer neurons then serve only to learn the missing non-linear part.
+ shared weights or sharing certain parts of the architecture (e.g. the first layer) between networks associated with several related tasks, can significantly improve generalization.
+ better to use a symmetric nonlinearity in the hidden layers (hyperbolic tangent, or *tanh*, unlike the logistic sigmoid), to avoid saturation of the nonlinearity in the hidden layers.<br/>
*Note: read improving the conditioning of the Hessian Matrix for accelerating gradient descent*

## Normalize Inputs

It is crucial that the inputs have a mean not too far from zero, and a variance not far from one. Values of the input must also not be too large in magnitude. You can perform certain monotonic, non-linear transformation that reduce large values to achieve this. If we have a very large input, it will saturate many neurons and block the learning algorithm for that example. The magnitudes (variances) of inputs to each layer must also be of the same order when using a single learning rate for all of the layers, in order to avoid one layer becoming a bottleneck (the slowest to train).

## Target Outputs

+ are always in the interval of values that the nonlinearity in the output layer can produce (and are roughly normal N(0, 1) in the linear case)
+ are not too near to the limits of the nonlinearity in the output layer: for classification, an optimal value is close to the two inflection points which is the points of maximal curvature/second derivative ($$-0.6$$ and $$0.6$$ for *tanh*, $$0.2$$ and $$0.8$$ for *sigmoid*).
+ best to use the cross-entropy criterion for probabilistic classification, or the “hinge” margin criterion (as is used for the perceptron and SVMs, but by penalizing departures from the decision surface beyond a margin).<br/>*Note: for a given $$x \in X$$, they assign probabilities to all $$y \in Y$$(and these probabilities sum to one)* $$\hat{y}=\text{argmax}_y\operatorname{Pr}(Y=y$$ l $$X)$$


---

Reference:
<br/><ins>http://cs231n.github.io/neural-networks-1/#quick</ins>
<br/><ins>http://www.iro.umontreal.ca/~bengioy/ift6266/H12/html/mlp_en.html</ins>
{: style="font-size: 80%; text-align: right; color: blue"}
