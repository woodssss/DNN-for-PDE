import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class tspnn:
    def __init__(self, layers, t0, x0, t_train, x_train, f0, v):
        self.layers = layers
        self.x0 = x0
        self.ti = t0
        self.f0 = f0

        self.t_train, self.x_train = t_train, x_train

        self.v = v
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])

        self.t_ip = tf.placeholder(tf.float32, shape=[None, self.x_train.shape[1]])
        self.x_ip = tf.placeholder(tf.float32, shape=[None, self.t_train.shape[1]])

        self.t0 = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])

        # tf graph

        self.pred = self.f_nn(self.x, self.t)

        self.f0_pred = self.f_nn(self.x, self.t0)

        self.pde = self.pde_nn(self.x_ip, self.t_ip)

        # loss

        self.loss = tf.reduce_mean(tf.square(self.f0_pred - self.f0)) + tf.reduce_mean(tf.square(self.pde))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        #self.optimizer_sgd = tf.keras.optimizers.Adam()
        #self.train_op_sgd = self.optimizer_sgd.minimize(self.loss)

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

    def f_nn(self, x, t):
        X = tf.concat([x,t],1)
        f= self.neural_net(X, self.weights, self.biases)
        return f

    def pde_nn(self, x, t):
        f = self.f_nn(x,t)
        f_x = tf.gradients(f, x)[0]
        f_t = tf.gradients(f, t)[0]
        pde = f_t + self.v* f_x
        return pde

    def train(self, nIter):

        tf_dict = {self.x: self.x0, self.t: self.ti, self.t0: 0*self.x0, self.t_ip: self.t_train,
                   self.x_ip: self.x_train}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            #self.sess.run(self.train_op_sgd, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        # self.optimizer.minimize(self.sess,
        #                         feed_dict=tf_dict,
        #                         fetches=[self.loss],
        #                         loss_callback=self.callback)

    def predict(self, x, t):

        tf_dict = {self.x: x, self.t: t}

        mypred = self.sess.run(self.pred, tf_dict)

        return mypred



if __name__ == "__main__":
    nx = 200
    lx = 2 * np.pi
    T = 1

    N = 500

    t_train = np.sort(np.random.rand(1, N) * T).T

    x_train = np.sort((np.random.rand(1, N)-0.5) * lx).T

    x0 = np.sort(np.random.rand(1, nx) * lx).T

    t0 = np.sort(np.random.rand(1, nx)).T

    T = 0.5 * np.ones((nx, 1))

    f0 = np.exp(-5 * (x0 - lx / 2) ** 2)

    xn = np.sort(np.random.rand(1, nx) * lx * 2).T

    layers = [2, 80, 80, 80, 1]

    v=1

    mdl = tspnn(layers, t0, x0, t_train, x_train, f0, v)

    mdl.train(10000)
    pd = mdl.predict(x0, T)
    plt.plot(x0, pd, x0, f0)
    plt.show()








