import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class tspnn:
    def __init__(self, layers, x0, dt, f0, v, q, RK4):
        self.layers = layers
        self.x0 = x0
        self.dt= dt
        self.RK4 = RK4
        self.f0 = f0
        self.q = q
        self.f0_rep = np.tile(self.f0, (1, self.q+1))
        self.v= v
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.f0_rep_tf = tf.placeholder(tf.float32, shape=[None, self.q+1])

        self.dummy_x1_tf = tf.placeholder(tf.float32, shape=[None, self.q])

        # tf graph

        self.pde_ic = self.pde_nn(self.x)

        self.pred = self.neural_net(self.x, self.weights, self.biases)

        # loss

        self.loss = tf.reduce_mean(tf.square(self.pde_ic - self.f0_rep_tf))

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

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

    def f_nn(self, x):
        f= self.neural_net(x, self.weights, self.biases)
        return f

    def mygrad(self, nn, x):
        g = tf.gradients(nn, x, grad_ys=self.dummy_x1_tf)[0]
        return tf.gradients(g, self.dummy_x1_tf)[0]

    def pde_nn(self, x):
        f_vec = self.neural_net(x, self.weights, self.biases)

        f_1_q = f_vec[:,:-1]

        f_x = self.mygrad(f_1_q, x)

        Lf = self.v * f_x

        print(Lf.shape, self.RK4.shape)

        f_ic = f_vec + self.dt*tf.matmul(Lf, self.RK4.T)

        return f_ic



    def train(self, nIter):

        tf_dict = {self.x: self.x0, self.f0_rep_tf: self.f0_rep, self.dummy_x1_tf: np.ones((self.x0.shape[0], self.q))}

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

    def predict(self, x):

        tf_dict = {self.x: x}

        mypred = self.sess.run(self.pred, tf_dict)

        return mypred



if __name__ == "__main__":
    nx = 300
    lx = 2 * np.pi

    # q is RKq method
    q=4

    # RK4 = np.array([[0, 0, 0, 0],
    #                 [0.5, 0, 0, 0],
    #                 [0, 0.5, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]])
    RK4 = np.array([[0.5, 0, 0, 0],
                    [1/6, 0.5, 0, 0],
                    [-0.5, 0.5, 0.5, 0],
                    [1.5, -1.5, 0.5, 0.5],
                    [1.5, -1.5, 0.5, 0.5]])

    RK4 = np.float32(RK4)

    print(type(RK4[0][0]))

    x0 = np.sort(np.random.rand(1, nx) * lx).T


    f0 = np.exp(-5 * (x0 - lx / 2) ** 2)

    xn = np.sort(np.random.rand(1, nx) * lx * 2).T

    layers = [1, 80, 80, 80, q+1]

    v=1

    dt=0.2

    mdl = tspnn(layers, x0, dt, f0, v, q, RK4)

    mdl.train(5000)
    pd = mdl.predict(x0)
    pd_cb = np.sum(pd[:, 0:4]* RK4[[-1], :], axis=1)
    #pd_cb = (tf.reduce_sum(pd[:, 0:4] * RK4[[-1], :], 1)[:, None]).numpy()

    print('shape', pd.shape, pd_cb.shape)

    plt.plot(x0, f0, '*r', x0, pd[:, -1], 'g')
    plt.show()








