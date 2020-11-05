import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class fp_chv_nn:
    def __init__(self, layers, x0, t0, g0):
        self.layers = layers
        self.x0 = x0
        self.ti = t0
        self.g0 = g0

        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])

        self.t0 = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])

        self.ones_M = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.trep = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        # tf graph

        self.pred = self.g_nn(self.x, self.t)

        self.g0_pred = self.g_nn(self.x, self.t0)

        self.pde = self.pde_nn(self.x, self.t)

        self.pred_at_t = self.g_nn(self.x, self.t)

        self.g_t_pred = self.g_t_nn(self.x, self.trep)



        # loss

        self.loss = tf.reduce_mean(tf.square(self.g0_pred - self.g0)) + tf.reduce_mean(tf.square(self.pde))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def callback(self, loss):
        print('Loss:', loss)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def g_nn(self, x, t):
        X = tf.concat([x,t],1)
        g = self.neural_net(X, self.weights, self.biases)
        return g

    def g_t_nn(self,x, t):

        g = self.g_nn(x, t)

        g_t = tf.gradients(g, t)[0]

        return g_t

    def get_I(self, tv):

        g = self.g_nn(self.x0, tv*np.ones_like(self.x0))

        gt = self.g_t_nn(self.x0, tv*np.ones_like(self.x0))

        weights_vec = tf.math.exp(-g)

        #print((tf.reduce_sum(weights_vec)).shape, weights_vec.shape)

        weights = weights_vec/tf.reduce_sum(weights_vec)

        I = tf.reduce_sum(gt*weights)

        print(I)

        return I

    def pde_nn(self, x, t):
        g = self.g_nn(x, t)

        g_x = tf.gradients(g, x)[0]
        g_t = tf.gradients(g, t)[0]
        g_xx = tf.gradients(g_x, x)[0]

        I = self.get_I(t)

        pde = g_t - x*g_x - g_xx + g_x**2 + np.ones_like(x) - I

        return pde

    def train(self, nIter):

        tf_dict = {self.x: self.x0, self.t: self.ti, self.t0: 0*self.x0, self.ones_M: np.ones_like(self.x0)}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x, t):

        tf_dict = {self.x: x, self.t: t}

        mypred = self.sess.run(self.pred, tf_dict)

        return mypred

    def predict_t(self, t):
        tf_dict = {self.x: self.x0, self.t: t*np.ones_like(self.x0)}

        g_pred_t = self.sess.run(self.pred, tf_dict)

        return g_pred_t



if __name__ == "__main__":
    nx = 300
    lx = 10

    x0 = np.sort((np.random.rand(1, nx)-0.5) * lx).T

    t0 = np.sort(np.random.rand(1, nx)).T

    T = 0.2* np.ones((nx, 1))

    f0 = np.exp(-5 * (x0-1) ** 2)*(np.pi/5)**(-0.5)

    print('mass', np.sum(f0)*lx/nx)

    g0 = 5 * x0 ** 2


    xn = np.sort(np.random.rand(1, nx) * lx * 2).T

    layers = [2, 100, 100, 1]

    v=1

    mdl = fp_chv_nn(layers, x0, t0, g0)

    mdl.train(20000)
    pd = mdl.predict(x0, 0*x0)

    npp = mdl.predict(x0, T)

    recover = np.exp(-npp)

    const = np.sum(recover)*lx/nx

    real_recover = recover/const
    plt.plot(x0, real_recover, x0, f0)
    plt.show()








