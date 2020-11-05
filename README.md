# Introduction
The application of deep nueral network in solving partial differential equation attracts tons of attentions nowaday. The most inportant advantage of DNN 
approach is to tackle the curse of dimentionality. Compared to regular numerical method, FD FV and FE for example, DNN approach does not require presetting 
the mesh grid of the computational domain; in fact, the data set is randomly sampled. In this project i will apply this approach to some interesting system.
# The error function and optimization methods

# Numerical examples
In this part, we consider following examples.
## Transport equation with zero b.c
Consider 1d transport equation f_t + v f_x = 0. The constraints consist of two parts: IC and PDE. Here 
we solve the transport equation by two ways: continuous time method and discrete time method. The main difference between these two methods is whether take time varibale t as input of NN.
## Continuous time approach
```
run tspnn.py
```

## Discrete time approach
```
run tspnndt.py
```
## Fokker Planck equation with zero b.c
Consider 
```
f_t = xf_x+f+f_xx 
```
Because this evolution equation preserve positivity and total mass, in order to ensure 
these two physics properties, one either uses two constraints in error function, or do change of variable. Here we implement in both ways.
### New error function
In addition to original error function, we add a new part required the mass conservation. For positivity, we simply use Softplus activation function at the last fully connected layer.
```
run fpnn.py
```
### Change of variable
Define f(t,x) = exp(-g(t,x))/c(t), where c(t) is a normalized constant that only depends on time. When recover f(t,x) from g(t,x), the expontial function ensure the positivity and division by the normalized constant ensure the mass conservation. And we now working with system g_t = xg_x + g_xx - g_x^2 -1 + I(t), here I(t) = \frac{\int g_t exp(-g) dx} {\int exp(-g) dx}. See detail in Reference.
```
run fp_chv_nn.py
```
Note that above two approaches are continuous time approachs, which is unlike to perform well on stiff system. Thus we introduce discrete time approach in following part, the stiffness problem can be resolved by leveraging the high order Runge-Kutta methods.



# Reference
[Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning](https://arxiv.org/pdf/1811.08782.pdf)\
[Physics Informed Neural Networks](https://github.com/maziarraissi/PINNs)


