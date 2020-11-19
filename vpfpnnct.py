import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

class vpfpnnct:
    def __init__(self, layers, layers2, tm, tm_ori, xm, xm_ori, vm, ntm, nxm, nvm, lx, lv, N_pair, eps, f_0_train, train_set):
        # This is PINN solver for VPFP system by continuous time method
        # Author: Wuzhe Xu
        # Date: 11/18/2020
        # tb, xb, vb are the boundary grid point, t_train, x_train, v_train are the collocation points
        # tm, xm, vm are used for calculating rho, generated randomly on x and v and then use meshgrid,
        # in contrast, train_set are generated totally randomly
        self.layers, self.layers2 = layers, layers2
        self.eps = eps
        self.lx, self.lv, self.N = lx, lv, N_pair

        self.tm, self.xm, self.vm, self.ntm, self.nxm, self.nvm = tm, xm, vm, ntm, nxm, nvm

        self.tm_ori, self.xm_ori = tm_ori, xm_ori

        self.ntxvm = self.ntm*self.nxm*self.nvm

        self.t_train = train_set[:, 0:1]
        self.x_train = train_set[:, 1:2]
        self.v_train = train_set[:, 2:3]

        self.f_0_train = f_0_train

        self.t_0_vec = np.zeros_like(self.t_train)

        self.x_lb_vec = -lx / 2 * np.ones_like(self.x_train)

        self.x_rb_vec = lx / 2 * np.ones_like(self.x_train)

        self.v_lb_vec = -lv / 2 * np.ones_like(self.v_train)

        self.v_rb_vec = lv / 2 * np.ones_like(self.v_train)

        self.weights, self.biases = self.initialize_NN(layers)

        self.weights2, self.biases2 = self.initialize_NN(layers2)

        # tf placeholder
        self.t_ip = tf.placeholder(tf.float32, shape=[None, self.t_train.shape[1]])
        self.x_ip = tf.placeholder(tf.float32, shape=[None, self.x_train.shape[1]])
        self.v_ip = tf.placeholder(tf.float32, shape=[None, self.v_train.shape[1]])

        self.tm_ip = tf.placeholder(tf.float32, shape=[self.ntxvm, self.tm.shape[1]])
        self.xm_ip = tf.placeholder(tf.float32, shape=[self.ntxvm, self.xm.shape[1]])
        self.vm_ip = tf.placeholder(tf.float32, shape=[self.ntxvm, self.vm.shape[1]])

        self.t_ori = tf.placeholder(tf.float32, shape=[self.tm_ori.shape[0], self.tm_ori.shape[1]])
        self.x_ori = tf.placeholder(tf.float32, shape=[self.xm_ori.shape[0], self.xm_ori.shape[1]])

        self.t_pred = tf.placeholder(tf.float32, shape=[None, self.t_train.shape[1]])
        self.x_pred = tf.placeholder(tf.float32, shape=[None, self.x_train.shape[1]])
        self.v_pred = tf.placeholder(tf.float32, shape=[None, self.v_train.shape[1]])

        self.t_0_ip = tf.placeholder(tf.float32, shape=[None, self.t_0_vec.shape[1]])

        self.x_lb_ip = tf.placeholder(tf.float32, shape=[None, self.x_lb_vec.shape[1]])

        self.x_rb_ip = tf.placeholder(tf.float32, shape=[None, self.x_rb_vec.shape[1]])

        self.v_lb_ip = tf.placeholder(tf.float32, shape=[None, self.v_lb_vec.shape[1]])

        self.v_rb_ip = tf.placeholder(tf.float32, shape=[None, self.v_rb_vec.shape[1]])

        # tf graph

        self.pde_vpfp = self.pde_vpfp_nn(self.t_ip, self.x_ip, self.v_ip)

        self.pde_poisson = self.pde_poisson_nn(self.t_ori, self.x_ori, self.tm_ip, self.xm_ip, self.vm_ip)

        self.pde_x_lb = self.f_nn(self.t_ip, self.x_lb_ip, self.v_ip)

        self.pde_x_rb = self.f_nn(self.t_ip, self.x_rb_ip, self.v_ip)

        self.pde_v_lb = self.f_nn(self.t_ip, self.x_ip, self.v_lb_ip)

        self.pde_v_rb = self.f_nn(self.t_ip, self.x_ip, self.v_rb_ip)

        self.f0_pred = self.f_nn(self.t_0_ip, self.x_ip, self.v_ip)

        self.f_pred = self.f_nn(self.t_pred, self.x_pred, self.v_pred)

        self.phi_pred = self.phi_nn(self.t_pred, self.x_pred)

        self.rho_pred = self.get_rho_nn(self.t_pred, self.x_pred, self.v_pred)

        self.mass = self.get_mass(self.tm_ip, self.xm_ip, self.vm_ip)

        self.mass_given = np.ones((self.ntm, 1))

        #self.pred_f = self.f_nn(self.xm_ip, self.vm_ip)

        # loss
        # loss 1: VPFP pde
        # loss 2: poisson equation
        # loss 3: I.C
        # loss 4: zero B.C on v direction
        # loss 5: periodic B.C on x
        # loss 6: conservation of total mass

        self.loss = tf.reduce_mean(tf.square(self.pde_vpfp)) + \
                    tf.reduce_mean(tf.square(self.pde_poisson)) + \
                    tf.reduce_mean(tf.square(self.f0_pred - self.f_0_train)) + \
                    tf.reduce_mean(tf.square(self.pde_v_lb)) + tf.reduce_mean(tf.square(self.pde_v_rb)) + \
                    tf.reduce_mean(tf.square(self.pde_x_lb - self.pde_x_rb)) + \
                    tf.reduce_mean(tf.square(self.mass - self.mass_given))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
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

    def f_nn(self, t, x, v):
        X = tf.concat([t, x, v], 1)
        f = self.neural_net(X, self.weights, self.biases)
        return f

    def f_nn_pred(self, t, x, v):

        tf_dict = {self.tm_ip: t, self.xm_ip: x, self.vm_ip: v}

        mypred = self.sess.run(self.f_pred, tf_dict)

        return mypred

    def phi_nn(self, t, x):
        X = tf.concat([t, x], 1)
        phi = self.neural_net(X, self.weights2, self.biases2)
        return phi

    def pde_vpfp_nn(self, t, x, v):

        f = self.f_nn(t, x, v)

        f_t = tf.gradients(f, t)[0]

        f_x = tf.gradients(f, x)[0]

        f_v = tf.gradients(f, v)[0]

        f_vv = tf.gradients(f_v, v)[0]

        phi = self.phi_nn(t, x)

        phi_x = tf.gradients(phi, x)[0]

        RHS = v*f_v + f + phi_x*f_v + f_vv

        pde = self.eps * (f_t + v*f_x) - RHS

        return pde

    def pde_poisson_nn(self, t_ori, x_ori, t, x, v):

        rho = self.get_rho_nn(t, x, v)

        lap_phi = self.lap_phi(t_ori, x_ori)

        h = self.h(x_ori)

        #rep = tf.constant([self.ntm, 1])

        #h_rep = tf.tile(h, rep)

        #print('s', lap_phi.shape, rho.shape, h_rep.shape)

        pde = lap_phi + rho - h

        return pde


    def lap_phi(self, t, x):
        phi = self.phi_nn(t, x)
        grad_phi = tf.gradients(phi, x)[0]
        lap_phi = tf.gradients(grad_phi, x)[0]
        return lap_phi

    def get_rho_nn(self, t, x, v):

        f = self.f_nn(t, x, v)

        sp = tf.ones([self.nvm, 1])

        sk = tf.linspace(np.float32(0), self.ntm*self.nxm - 1, self.nxm*self.ntm, name="linspace")

        sk = tf.reshape(sk, (self.nxm*self.ntm, 1))

        id = tf.contrib.kfac.utils.kronecker_product(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.nxm * self.nvm * self.ntm])

        print('sid', id.shape, f.shape)

        rho = tf.math.segment_sum(f, id)*self.lv/self.nvm

        return rho

    def h(self, x):

        h = 5.0132/1.2661 * tf.exp(tf.cos(np.pi*2*x))

        return h

    def get_mass(self, t, x, v):

        print('now getting mass')

        rho = self.get_rho_nn(t, x, v)

        sp = tf.ones([self.nxm, 1])

        sk = tf.linspace(np.float32(0), self.ntm  - 1, self.ntm, name="linspace")

        sk = tf.reshape(sk, (self.ntm, 1))

        id = tf.contrib.kfac.utils.kronecker_product(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.nxm * self.ntm])

        M = tf.math.segment_sum(rho, id) * self.lx / self.nxm

        return M

    def train(self, nIter):

        tf_dict = {self.t_ip: self.t_train, self.x_ip: self.x_train, self.v_ip: self.v_train, self.tm_ip: self.tm,
                   self.xm_ip: self.xm, self.vm_ip: self.vm, self.t_ori: self.tm_ori, self.x_ori: self.xm_ori,
                   self.t_0_ip: self.t_0_vec, self.v_lb_ip: self.v_lb_vec, self.v_rb_ip: self.v_rb_vec,
                   self.x_lb_ip: self.x_lb_vec, self.x_rb_ip: self.x_rb_vec}

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
                                loss_callback= None)
        # loss_callback=self.callback)

    def phi_predict(self, t, x):

        tf_dict = {self.t_pred: t, self.x_pred: x}

        mypd = self.sess.run(self.phi_pred, tf_dict)

        return mypd

    def rho_predict(self, t, x, v):

        tf_dict = {self.t_pred: t, self.x_pred: x, self.v_pred: v}

        mypred = self.sess.run(self.rho_pred, tf_dict)

        return mypred

    def predict(self, t, x, v):

        print('doing prediction')

        tf_dict = {self.t_pred: t, self.x_pred: x, self.v_pred: v}

        mypred = self.sess.run(self.f_pred, tf_dict)

        return mypred


if __name__ == "__main__":
    ntm = 20
    nxm = 20
    nvm = 200
    lx = 1
    lv = 10
    T = 1

    N_pair = 5000

    tm = np.linspace(0, T, ntm).T[:, None]

    tm_ori = tm

    t_d_ori = tm

    xm = np.linspace(-lx / 2, lx / 2, nxm).T[:, None]

    xm_ori = xm

    x_d_ori = xm

    rho0 = 0.5*(2+np.cos(np.pi*2*xm))

    vm = np.linspace(-lv / 2, lv / 2, nvm).T[:, None]

    vm_ori = vm

    v_d_ori = vm

    res = np.ones((nxm * nvm, 2))

    for i in range(xm.shape[0]):
        res[i * nvm:(i + 1) * nvm, [0]] = xm[i][0] * np.ones_like(vm)
        res[i * nvm:(i + 1) * nvm, [1]] = vm

    data_set = np.ones((nxm * nvm* ntm, 3))

    for i in range(ntm):
        data_set[i * nvm * nxm:(i + 1) * nvm * nxm, 0:1] = tm[i][0] * np.ones((nxm * nvm, 1))
        data_set[i * nvm * nxm:(i + 1) * nvm * nxm, 1:3 ] = res

    data_ori = np.ones((nxm * ntm, 2))

    for i in range(ntm):
        data_ori[i * nxm: (i+1)*nxm, 0:1] = tm[i][0] * np.ones((nxm, 1))
        data_ori[i * nxm: (i + 1) * nxm, 1:2] = xm

    tm_ori = np.float32(data_ori[:, 0:1])

    xm_ori = np.float32(data_ori[:, 1:2])

    tm = np.float32(data_set[:, 0:1])

    xm = np.float32(data_set[:, 1:2])

    vm = np.float32(data_set[:, 2:3])

    t_train = ((np.random.rand(1, N_pair)) * T).T

    x_train = ((np.random.rand(1, N_pair) - 0.5) * lx).T

    v_train = ((np.random.rand(1, N_pair) - 0.5) * lv).T

    f0_temp = 0.5*(2+np.cos(np.pi*2*x_train)) * np.exp(-0.5 * (v_train - 0.5) ** 2) * (np.pi / 0.5) ** (-0.5)

    f_0_train = np.float32(f0_temp)

    print('mass = ', np.sum(f_0_train) / N_pair * lx * lv)

    train_set = np.float32(np.concatenate((t_train, x_train, v_train), axis=1))

    xxm, vvm = np.meshgrid(x_d_ori, v_d_ori)

    # res = np.ones((nxm * nvm, 2))
    # for i in range(xm.shape[0]):
    #     res[i * nvm:(i + 1) * nvm, [0]] = xm[i][0] * np.ones_like(vm)
    #     res[i * nvm:(i + 1) * nvm, [1]] = vm

    # Take x,v as input, ouput f star, f bar and phi
    layers = [3, 120, 120, 120, 1]

    layers2 = [2, 60, 60, 1]

    eps = 1

    mdl = vpfpnnct(layers, layers2, tm, tm_ori, xm, xm_ori, vm, ntm, nxm, nvm, lx, lv, N_pair, eps, f_0_train, train_set)

    mdl.train(10000)
    pd = mdl.predict(tm, xm, vm)

    print('pd shape', pd.shape)

    pd_m = (np.reshape(pd[0:nxm*nvm, :], (nxm, nvm))).T

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    print('surf', xxm.shape, vvm.shape, pd_m.shape)

    # surf = ax.scatter3D(x_train, v_train, pd)
    surf = ax.plot_surface(xxm, vvm, pd_m)

    # plt.plot_surface(xx, vv, f_p)
    # plt.plot(x0, pd[:,-1], x0, f0)
    plt.show()

    #phi_pd, _, _ = mdl.phi_predict(xm_ori)

    phi_pd = mdl.phi_predict(tm_ori, xm_ori)

    rho_pd = mdl.rho_predict(tm, xm, vm)

    print('shape', phi_pd.shape, rho_pd.shape, xm.shape, vm.shape)

    #phi_pd = phi_pd[0]

    rho_pd0 = rho_pd[0:nxm]

    rho_pd_mid = rho_pd[(np.int32(ntm/2)-1)*nxm: np.int32(ntm/2)*nxm]

    rho_pd_end = rho_pd[(ntm-1)*nxm :]

    #print('ssss', rho_pd0.shape, rho_pd_mid.shape, rho_pd_end.shape)

    plt.plot(x_d_ori, rho_pd0, 'r', x_d_ori, rho_pd_mid, 'k', x_d_ori, rho_pd_end, 'b *', x_d_ori, rho0, 'g')
    plt.gca().legend(('rho0_pred', 'rho_zp5', 'rho_1', 'exact_rho0'))
    plt.show()








