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
f_t = 1/eps (xf_x+f+f_xx) 
```
Because this evolution equation preserve positivity and total mass, in order to ensure 
these two physics properties, one either uses two constraints in error function, or do change of variable. Here we implement in both ways.
### New error function
In addition to original error function, we add a new part required the mass conservation. For positivity, we simply use Softplus activation function at the last fully connected layer.
```
run fpnn.py
```
Notice that this method works quite well on convergence to equilibrium. By using small eps, such as eps = 0.01, 0.001, 0.0001, the neural net gives 
good approximation to equilibrium by plugging in t=0.1.
### Change of variable
Define f(t,x) = exp(-g(t,x))/c(t), where c(t) is a normalized constant that only depends on time. When recover f(t,x) from g(t,x), the expontial function ensure the positivity and division by the normalized constant ensure the mass conservation. And we now working with system g_t = xg_x + g_xx - g_x^2 -1 + I(t), here I(t) = \frac{\int g_t exp(-g) dx} {\int exp(-g) dx}. See detail in Reference.
```
run fp_chv_nn.py
```
Note that above two approaches are continuous time approachs. Beside, we introduce discrete time approach in following part, the stiffness problem can be resolved by leveraging the high order Runge-Kutta methods.
### Discrete time method
In this part, we consider a modified system: to see the long time behavior, we scale time variable t by t` = \eps t, then we have following new system
```
f_t = 1/eps (xf_x+f+f_xx) 
```
This is a stiff system as eps goes to 0. To resolve the stiffness, we use 100 stage implicit Runge Kutta method, and let dt = 0.1, eps = 0.05, we see approximation at t=0.1 matches the analytical equilibrium very well.
```
run fpnndt.py
```
However, if we use even smaller eps, the convergence rate slows down dramatically. The is caused by the appearance of eps in the error function, which significantly slow down the convergence process. We will try to accelerate the convergence process from optimization aspect.

# VPFP system
Now let us consider VPFP system, i.e
```
f_t + v f_x = 1/eps (vf_v + phi_x f_v + f + f_vv)
```
And phi(t,x) satisfies
```
phi_xx = \int f dv - h(x)
```
In this example, we use two separate NN, the first one is to approximate vlasov-fokker-planck equation and the other one is to for the poisson equation. The error function contains 6 parts: 1. error of VFP; 2. error of poisson equation; 3. error of initial condition; 4. zero boundary condition on v direction; 5. periodic B.C on x; 6. conservation of total mass.
```
run vpfpnnct.py
```


# Reference
[Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning](https://arxiv.org/pdf/1811.08782.pdf)\
[Physics Informed Neural Networks](https://github.com/maziarraissi/PINNs)


