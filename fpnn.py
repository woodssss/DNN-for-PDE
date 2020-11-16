import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class fpnn:
    def __init__(self, layers, x0, t0, f0, lx, nx, eps, train_set):
        self.layers = layers
        self.x0 = x0
        self.ti = t0
        self.f0 = f0
        self.lx = lx
        self.nx = nx

        self.xts = train_set[:, 0:1]
        self.tts = train_set[:, 1:2]

        self.mylb = np.ones_like(self.x0)*(-lx/2)

        self.myrb = np.ones_like(self.x0) * (lx / 2)

        self.eps = eps

        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.x_train = tf.placeholder(tf.float32, shape=[None, self.xts.shape[1]])

        self.xlb = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.xrb = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.t = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])

        self.t_train = tf.placeholder(tf.float32, shape=[None, self.tts.shape[1]])

        self.t0 = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])

        # tf graph

        self.pred = self.f_nn(self.x, self.t)

        self.f0_pred = self.f_nn(self.x, self.t0)

        self.predbcl = self.f_nn(self.xlb, self.t)

        self.predbcr = self.f_nn(self.xrb, self.t)

        self.pde = self.pde_nn(self.x_train, self.t_train)

        self.mass = self.get_I(self.t)

        self.m_given = np.ones_like(self.f0)

        # loss

        self.loss = tf.reduce_mean(tf.square(self.f0_pred - self.f0)) + \
                    tf.reduce_mean(tf.square(self.pde)) + \
                    tf.reduce_mean(tf.square(self.mass-self.m_given)) + tf.reduce_mean(tf.square(self.predbcl)) + \
                    tf.reduce_mean(tf.square(self.predbcr))


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 10**-5 })
                                                                         #'ftol': 1.0 * np.finfo(float).eps})

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
        Y = tf.nn.softplus(Y)
        return Y

    def f_nn(self, x, t):
        X = tf.concat([x,t],1)
        f= self.neural_net(X, self.weights, self.biases)
        return f

    def f_t_nn(self,x, t):

        f = self.f_nn(x, t)

        f_t = tf.gradients(f, t)[0]

        return f_t

    # def get_I(self, tv):
    #
    #     ft = self.f_t_nn(self.x0, tv*np.ones_like(self.x0))
    #
    #     I = tf.reduce_sum(ft)*self.lx/self.nx
    #
    #     return I

    def get_I(self, tv):

        f = self.f_nn(self.x0, tv*np.ones_like(self.x0))

        I = tf.reduce_sum(f)*self.lx/self.nx

        return I

    # def pde_nn(self, x, t):
    #     f = self.f_nn(x, t)
    #     f_x = tf.gradients(f, x)[0]
    #     f_t = tf.gradients(f, t)[0]
    #     f_xx = tf.gradients(f_x, x)[0]
    #
    #     pde = f_t - 1/self.eps*(f - x*f_x - f_xx)
    #     return pde

    def pde_nn(self, x, t):
        f = self.f_nn(x, t)
        f_t = tf.gradients(f, t)[0]
        f_x = tf.gradients(f, x)[0]
        f_xx = tf.gradients(f_x, x)[0]

        pde = self.eps*f_t - (f + x*f_x + f_xx)
        return pde

    def train(self, nIter):

        tf_dict = {self.x: self.x0, self.t: self.ti, self.t0: 0*self.x0,
                   self.xlb: self.mylb, self.xrb: self.myrb, self.x_train: self.xts, self.t_train: self.tts}

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



if __name__ == "__main__":
    nx = 400
    nt = nx
    lx = 10

    N_pair = 800

    x0 = np.sort((np.random.rand(1, nx)-0.5) * lx).T

    x_train = ((np.random.rand(1, N_pair)-0.5) * lx).T

    t0 = np.sort(np.random.rand(1, nx)).T

    t_train = ((np.random.rand(1, N_pair))).T

    train_set = np.concatenate((x_train, t_train), axis=1)

    T = 0.5 * np.ones((nt, 1))

    T0 = 0 * np.ones((nt, 1))

    f0 = np.exp(-1 * (x0) ** 2)*(np.pi/1)**(-0.5)

    fexact = np.exp(-0.5 * (x0) ** 2)*(np.pi*2)**(-0.5)

    xn = np.sort(np.random.rand(1, nx) * lx * 2).T

    layers = [2, 200, 200, 200, 200, 1]

    v=1

    eps = 0.005

    mdl = fpnn(layers, x0, t0, f0, lx, nx, eps, train_set)

    mdl.train(15000)
    pd = mdl.predict(x0, T)
    pd0 = mdl.predict(x0, T0)
    plt.plot(x0, pd, 'r', x0, f0, 'g', x0, fexact, 'b*', x0, pd0, 'k')
    plt.gca().legend(('approx f(T,x)', 'f(0,x)', 'Equilibrium', 'approx f(0,x)'))
    #plt.title('eps = ', eps)
    plt.show()








