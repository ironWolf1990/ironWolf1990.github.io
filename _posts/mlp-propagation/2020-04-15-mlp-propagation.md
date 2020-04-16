---
title: Forward and Back Propagation
layout: post
category: Deep Learning
date: 2020-04-15
excerpt: "A multilayer perceptron is a deep, artificial neural network. It is composed of more than one perceptron."
abstract: "A multilayer perceptron is a deep, artificial neural network. It is composed of more than one perceptron. They are composed of an input layer to receive the signal, an output layer that makes a decision or prediction about the input, and in between those two, an arbitrary number of hidden layers that are the true computational engine of the MLP. MLPs with one hidden layer are capable of approximating any continuous function."
---

## Multilayer Perceptron (MLP)

Multilayer perceptrons are often applied to supervised learning problems: they train on a set of input-output pairs and learn to model the correlation (or dependencies) between those inputs and outputs. Training involves adjusting the parameters, or the weights and biases, of the model in order to minimize error. Backpropagation is used to make those weigh and bias adjustments relative to the error, and the error itself can be measured in a variety of ways, including by root mean squared error.

### Gradient Descent

One of the biggest advantages of the activation function over the unit step function is that it is differentiable. This property allows us to define a cost function $$J(\mathbf{w})$$ that we can minimize in order to update our weights. In the case of the linear activation function, we can define the cost function $$J(\mathbf{w})$$ as the sum of squared errors (SSE), which is similar to the cost function that is minimized in ordinary least squares (OLS) linear regression.
\begin{equation}
\operatorname{J}(w)=\frac{1}{2}\sum_{i=0}^{n}(target_i−output_i)^2
\end{equation} 

The principle behind gradient descent as **climbing down a hill** until a local or global minimum is reached. At each step, we take a step into the opposite direction of the gradient, and the step size is determined by the value of the learning rate as well as the slope of the gradient.

1. Each update is updated by taking a step into the opposite direction of the gradient
\begin{equation}
\Delta w = − \eta\text{ }∇\operatorname{J}(w)
\end{equation} 

2. Compute the partial derivative of the cost function for each weight in the weight vector
\begin{equation}
\Delta w_j = - \eta \frac{\partial J}{\partial w_j}
\end{equation}

3. Partial derivative of the SSE cost function for a particular weight can be calculated as
\begin{align}
\Delta w_j & = - \eta \sum_{i} (t^{(i)} - o^{(i)})(- x^{(i)}_{j}) \\\\ & = \eta \sum_i (t^{(i)} - o^{(i)})(x^{(i)}_j)
\end{align}

4. We can apply a simultaneous weight update similar to the perceptron rule:
\begin{equation}
w=w+\Delta{w}
\end{equation}

![Perceptron Gradient Descent]({{site.url}}{{site.baseurl}}{{site.assets_path}}/img/mlp-propagation/perceptron_gradient_descent.png?style=center "Perceptron Gradient Descent")

### Learning Rule

The learning rule above looks identical to the perceptron rule, the two main differences:
  1. output **o** is a real number and not a class label as in the perceptron learning rule.
  2. weight update is calculated based on all samples in the training set (instead of updating the weights incrementally after each sample), which is why this approach is also called **batch gradient descent**.

If the learning rate is too large, gradient descent will overshoot the minima and diverge. If the learning rate is too small, the algorithm will require too many epochs to converge and can become trapped in local minima more easily.

![Perceptron Learning Rate]({{site.url}}{{site.baseurl}}{{site.assets_path}}/img/mlp-propagation/perceptron_learning_rate.png?style=center "Perceptron Learning Rate")

It is easier to find an appropriate learning rate if the features are on the same scale, but it also often leads to faster convergence and can prevent the weights from becoming too small. A common way of feature scaling is standardization

$${x}_{j, std} = \frac{x_j - \mu_j}{\sigma_j}$$

where 
+ $$\mu_j$$ is the sample mean of the feature $$x_{j}$$ 
+ $$\sigma_j$$ the standard deviation

## Training MLP

In neural networks, you forward propagate to get the output and compare it with the real value to get the error. To minimize the error, you propagate backwards by finding the derivative of error with respect to each weight and then subtracting this value from the weight value.

The basic learning that has to be done in neural networks is training neurons when to get activated. Each neuron should activate only for particular type of inputs and not all inputs. Therefore, by propagating forward you see how well your neural network is behaving and find the error. After you find out that your network has error, you back propagate and use a form of gradient descent to update new values of weights. Then, you will again forward propagate to see how well those weights are performing and then will backward propagate to update the weights. This will go on until you reach some minima for error value.

![MLP Training]({{site.url}}{{site.baseurl}}{{site.assets_path}}/img/mlp-propagation/mlp-training.png?style=center "MLP Training")

+ $$n_l$$ denotes the number of layers in our network; thus $$n_l=3$$
+ neural network has parameters $$(W,b)$$, where $$W_{ij}^{(l)}$$ denotes the parameter (or weight) associated with the connection between unit $$j$$ in layer $$l$$ and unit $$i$$ in layer $$l+1$$;<br/> $$b_{i}^{(l)}$$ is the bias associated with unit $$i$$ in layer $$l+1$$.
+ bias units don’t have inputs or connections going into them, since they always output the value +$$1$$; let $$s_l$$ denote the number of nodes in layer $$l$$ not counting the bias unit.
+ $$a_{i}^{(l)}$$ denotes the activation (meaning output value) of unit $$i$$ in layer $$l$$ (for $$l=1$$, $$a_{i}^{(1)}=x_i$$ denotes the $$i^{\text{th}}$$ input)

### Forward Propagation

Given the parameters defined above, the computation that the network represents is given by:

1. At $$L_2$$, hidden layer :
\begin{align}
a_{1}^{(2)} & = f(W_{11}^{(1)}x_1 + W_{12}^{(1)} x_2 + W_{13}^{(1)} x_3 + b_1^{(1)}) \\\\ a_2^{(2)} & = f(W_{21}^{(1)}x_1 + W_{22}^{(1)} x_2 + W_{23}^{(1)} x_3 + b_2^{(1)}) \\\\ a_3^{(2)} & = f(W_{31}^{(1)}x_1 + W_{32}^{(1)} x_2 + W_{33}^{(1)} x_3 + b_3^{(1)})
\end{align}
>bias units don’t have inputs or connections going into them, since they always output the value +1.

2. At $$L_3$$, output neuron :
\begin{align}
h_{W,b}(x) & = a_1^{(3)} \\\\ h_{W,b}(x) & =  f(W_{11}^{(2)}a_1^{(2)} + W_{12}^{(2)} a_2^{(2)} + W_{13}^{(2)} a_3^{(2)} + b_1^{(2)})
\end{align}
>neural network defines a hypothesis $$\operatorname{h_{W,b}}(x)$$, for fixed setting of the parameters $$(W,b)$$, that outputs a real number.

3. let $$z_{i}^{(l)}$$ denote the total weighted sum of inputs to unit $$i$$ in layer $$l$$, including the bias term :
\begin{align}
z_{i}^{2} & = \sum_{j=1}^{n}(W_{ij}^{1}x_j + b_{i}^{1}) \\\\ a^{(l)}_i & = f(z^{(l)}_i)
\end{align}

4. extending the activation function $$\operatorname{f}(⋅)$$ to apply to vectors in an element-wise fashion :
\begin{align}
z^{(2)} &= W^{(1)} x + b^{(1)} \\\\ a^{(2)} & = f(z^{(2)}) \\\\ z^{(3)} & = W^{(2)} a^{(2)} + b^{(2)} \\\\ h_{W,b}(x) & = a^{(3)} = f(z^{(3)})
\end{align}
>by organizing our parameters in matrices and using matrix-vector operations, we can take advantage of fast linear algebra routines to quickly perform calculations in our network

### Backpropagation

Backward Propagation of Errors, often abbreviated as BackProp is one of the several ways in which an artificial neural network (ANN) can be trained. It is a supervised training scheme, which means, it learns from labeled training data (there is a supervisor, to guide its learning). BackProp is like **learning from mistakes**, corrects the ANN whenever it makes mistakes. An ANN consists of nodes in different layers; input layer, intermediate hidden layer(s) and the output layer. The connections between nodes of adjacent layers have **weights** associated with them. The goal of learning is to assign correct weights for these edges. Given an input vector, these weights determine what the output vector is. In supervised learning, the training set is labeled. This means, for some given inputs, we know the desired/expected output (label).

+ Given a training example $$(x,y)$$ , we will first run a **forward pass** to compute all the activations throughout the network, including the output value of the hypothesis $$h_{W,b}(x)$$.
+ for each node $$i$$ in layer $$l$$, compute an **error term** $$δ^{(l)}_{i}$$ that measures how much that node was **responsible** for any errors in our output.
+ for output node, we can directly measure the difference between the network’s activation and the true target value, and use that to define $$δ^{(n_l)}_{i}$$ (where layer $$n_l$$ is the output layer).
+ for hidden units, $$δ_{i}^{(l)}$$ is computed based on a weighted average of the error terms of the nodes that uses $$a_{i}^{(l)}$$ as an input.

Back prop is just gradient descent on individual errors. You compare the predictions of the neural network with the desired output and then compute the gradient of the errors with respect to the weights of the neural network. This gives you a direction in the parameter weight space in which the error would become smaller. In detail, the backpropagation algorithm:

1. Perform a feedforward pass, computing the activations for layers $$L_2$$, $$L_3$$, and so on up to the output layer $$L_{n_l}$$.

2. For each output unit $$i$$ in layer $$n_l$$ (the output layer), set :
\begin{align}
\delta^{(n_l)}_i &= \frac{\partial}{\partial z^{(n_l)}_i} \;\;\frac{1}{2} \left\| y - h\_{W,b} (x)\right\|^2 \\\\ &= - (y_i - a^{(n_l)}_i) \cdot f'(z^{(n_l)}_i)
\end{align}

3. each node $$i$$ in layer $$l$$ (where $$l = n_l-1, n_l-2, n_l-3, \ldots$$), set :
\begin{align}
\delta^{(l)}\_i = \left( \sum_{j=1}^{s_{l+1}} W^{(l)}_{ji} \delta^{(l+1)}_j \right) f'(z^{(l)}_i)
\end{align}

4. Compute the desired partial derivatives, which are given as:
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} \;\; J(W,b; x, y) & = a^{(l)}_j \delta_i^{(l+1)} 
\end{align}

\begin{align}
\frac{\partial}{\partial b_{i}^{(l)}} \;\; J(W,b; x, y) & = \delta_i^{(l+1)}
\end{align}

To train our neural network repeatedly take steps of gradient descent to reduce our cost function $$\operatorname{J}(W,b)$$. Implementing one batch of gradient decent, where $$ΔW^{(l)}$$ is a matrix (of the same dimension as $$W^{(l)}$$), and $$Δb^{(l)}$$ is a vector (of the same dimension as b(l)) :

1. Set $$\Delta W^{(l)}=0 \;\; \Delta b^{(l)}=0$$ (matrix/vector of zeros) for all $$l$$.

2. From $$i=1$$ to $$m$$,
    1. Use backpropagation to compute $$\nabla_{W^{(l)}} J(W,b;x,y)$$ and $$\nabla_{b^{(l)}} J(W,b;x,y)$$

    2. Set $$\Delta W^{(l)}=\Delta W^{(l)} + \nabla_{W^{(l)}} J(W,b;x,y)$$

    3. Set $$\Delta b^{(l)}=\Delta b^{(l)} + \nabla_{b^{(l)}} J(W,b;x,y)$$

3. Update parameters,
\begin{align}
W^{(l)} &= W^{(l)} - \alpha \left[ \left(\frac{1}{m} \Delta W^{(l)} \right) + \lambda W^{(l)}\right] \\\\ b^{(l)} &= b^{(l)} - \alpha \left[\frac{1}{m} \Delta b^{(l)}\right]
\end{align}
>where $$\alpha$$ is the learning rate parameter and $$\lambda$$ is called the weight decay parameter.

---

Reference:
<br/><ins>http://deeplearning.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/</ins>
<br/><ins>http://www.iro.umontreal.ca/~bengioy/ift6266/H12/html/mlp_en.html</ins>
{: style="font-size: 100%; text-align: left; color: blue;"}