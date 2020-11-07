import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


class fpnndt:
    def __init__(self, layers, x0, dt, f0, q, lx, nx, eps):
        self.layers = layers
        self.x0 = x0
        self.dt= dt
        self.f0 = f0
        self.q = q
        self.nx = nx
        self.lx = lx
        self.xbc = np.array([[-lx/2], [lx/2]])
        self.eps = eps
        self.v = 1
        self.f0_rep = np.tile(self.f0, (1, self.q+1))
        self.weights, self.biases = self.initialize_NN(layers)

        self.path = os.getcwd()

        tmp = np.float32(np.loadtxt('D:\DNNPDE\IRK100.txt'))
        self.IRK_weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))
        self.RK = self.IRK_weights
        self.IRK_times = tmp[q ** 2 + q:]


        # tf placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.xb = tf.placeholder(tf.float32, shape=[self.xbc.shape[0], self.xbc.shape[1]])

        self.f0_rep_tf = tf.placeholder(tf.float32, shape=[None, self.q+1])

        self.dummy_x1_tf = tf.placeholder(tf.float32, shape=[None, self.q])

        # tf graph

        self.pde_ic = self.pde_nn(self.x)

        self.pred = self.neural_net(self.x, self.weights, self.biases)

        self.predbc = self.neural_net(self.xb, self.weights, self.biases)

        self.mass_pred = self.get_mass(self.x)

        self.mass = np.ones((1, self.q+1))

        #self.mass = np.array([[1]])

        # loss

        self.loss = tf.reduce_mean(tf.square(self.pde_ic - self.f0_rep_tf)) + \
                    tf.reduce_mean(tf.square(self.mass_pred - self.mass)) + \
                    tf.reduce_mean(tf.square(self.predbc))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                #method='BFGS',
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
        Y = tf.nn.softplus(Y)
        #Y = tf.nn.sigmoid(Y)
        #Y = self.gelu(Y)
        #Y = self.gaussian_act(Y)
        #Y = self.mytanh_act(Y)
        #Y = tf.nn.tanh(Y) + np.ones_like(Y)
        return Y

    def gelu(self, x):
        """Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
            x: float Tensor to perform activation.
        Returns:
            `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    def gaussian_act(self, x):
        return tf.math.exp(-tf.pow(x,2))

    def mytanh_act(self, x):
        return tf.math.tanh(x) + np.ones_like(x)

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

        f_xx = self.mygrad(f_x, x)

        Lf = (f_x*x + f_1_q + f_xx)/self.eps
        #Lf = self.v*f_x

        f_ic = f_vec - self.dt*tf.matmul(Lf, self.RK.T)

        return f_ic

    def get_mass(self, x):
        f_ic = self.pde_nn(x)
        m_vec = tf.reduce_sum(f_ic, 0)*self.lx/self.nx
        print('sshp', f_ic.shape, m_vec.shape)
        return m_vec

    # def get_mass(self, x):
    #     f_ic = self.pde_nn(x)
    #     m_vec = tf.reduce_sum(f_ic[:,-1])*self.lx/self.nx
    #     return m_vec



    def train(self, nIter):

        tf_dict = {self.x: self.x0, self.f0_rep_tf: self.f0_rep, self.dummy_x1_tf: np.ones((self.x0.shape[0], self.q)),
                   self.xb: self.xbc}

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

        # self.optimizer.minimize(self.sess,
        #                         feed_dict=tf_dict,
        #                         fetches=[self.loss],
        #                         loss_callback=self.callback)
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss])

    def predict(self, x):

        tf_dict = {self.x: x}

        mypred = self.sess.run(self.pred, tf_dict)

        return mypred



if __name__ == "__main__":
    nx = 200
    lx = 10

    # q is RKq method
    q=100

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

    x0 = np.sort((np.random.rand(1, nx)-0.5) * lx).T


    f0 = np.exp(-2 * (x0) ** 2)*(np.pi/2)**(-0.5)

    f_exact = np.exp(-0.5 * (x0) ** 2)*(np.pi*2)**(-0.5)

    xn = np.sort(np.random.rand(1, nx) * lx * 2).T

    layers = [1, 100, 100, 100, q+1]

    v=1

    dt=0.1

    eps = 0.05

    mdl = fpnndt(layers, x0, dt, f0, q, lx, nx, eps)

    mdl.train(20000)
    pd = mdl.predict(x0)
    plt.plot(x0, pd[:,-1], 'b-', x0, f0, '*r', x0, f_exact , 'g')
    plt.gca().legend(('approx', 'ic', 'EQ'))
    plt.show()








