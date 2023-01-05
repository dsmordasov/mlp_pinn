import tensorflow as tf
# =============================================================================
# Do you know how much trouble it took to create a conda environment with
# a working tensorflow 1.15 version??? Just everything that could be wrong in
# the process went wrong, it took me too much effort to solve this completely
# =============================================================================
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
from scipy import integrate


# set seed for reproduce results
np.random.seed(1234)
tf.set_random_seed(1234)




class PhysicsInformedNN:
    # --------------------------------------------------------------------------------------------
    # Train up a NN_u and NN_f concurrently 
    # --------------------------------------------------------------------------------------------
    def __init__(self, X_u, X_f, layers, lower_bounds, upper_bounds, lambda_m, lambda_d, lambda_b):

        # --------------------------------------------------------------------------------------------
        # We can bound the domain based on the underlying physics of the system
        # We also initialise the system with its known parameters - mass, damping and connectivity
        # --------------------------------------------------------------------------------------------
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.layers = layers
        self.lambda_m = lambda_m
        self.lambda_d = lambda_d
        self.lambda_b = lambda_b

        # --------------------------------------------------------------------------------------------
        # We initialise with an input vector X_u of training data
        # --------------------------------------------------------------------------------------------
        self.time_u = X_u[:, 0:1]
        self.power_u = X_u[:, 1:2]
        self.delta_0_u = X_u[:, 2:3]
        self.omega_0_u = X_u[:, 3:4]

        # --------------------------------------------------------------------------------------------
        # And with the input vector X_f of collocation points (i.e. physics enforcement)
        # --------------------------------------------------------------------------------------------
        self.time_f = X_f[:, 0:1]
        self.power_f = X_f[:, 1:2]
        self.delta_0_f = X_f[:, 2:3]
        self.omega_0_f = X_f[:, 3:4]

        # --------------------------------------------------------------------------------------------
        # Run the modified NN initialisation, do some tensorflow magic (sess?), create tensor placeholders
        # --------------------------------------------------------------------------------------------
        self.weights, self.biases = self.initialize_NN(layers)
        
        #tf.compat.v1.disable_eager_execution()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        

        self.time_u_tf = tf.placeholder(tf.float32, shape=[None, self.time_u.shape[1]])
        self.power_u_tf = tf.placeholder(tf.float32, shape=[None, self.power_u.shape[1]])
        self.delta_0_u_tf = tf.placeholder(tf.float32, shape=[None, self.delta_0_u.shape[1]])
        self.omega_0_u_tf = tf.placeholder(tf.float32, shape=[None, self.omega_0_u.shape[1]])

        self.time_f_tf = tf.placeholder(tf.float32, shape=[None, self.time_f.shape[1]])
        self.power_f_tf = tf.placeholder(tf.float32, shape=[None, self.power_f.shape[1]])
        self.delta_0_f_tf = tf.placeholder(tf.float32, shape=[None, self.delta_0_f.shape[1]])
        self.omega_0_f_tf = tf.placeholder(tf.float32, shape=[None, self.omega_0_f.shape[1]])



        self.u_pred, self.u_t_pred, = self.pyhsics_net(self.time_u_tf,
                                                       self.power_u_tf,
                                                       self.delta_0_u_tf,
                                                       self.omega_0_u_tf)[0:2]

        self.f_pred = self.pyhsics_net(self.time_f_tf,
                                       self.power_f_tf,
                                       self.delta_0_f_tf,
                                       self.omega_0_f_tf)[2:3]

        # --------------------------------------------------------------------------------------------
        # Loss function as the MSE of the solution and its derivative, but also MSE_f which enforces the physics
        # f should be as close to 0 as possible for the dynamical system as per (4)
        # --------------------------------------------------------------------------------------------
        self.loss = (tf.reduce_mean(tf.square(self.delta_0_u_tf - self.u_pred)) +
                     tf.reduce_mean(tf.square(self.omega_0_u_tf - self.u_t_pred)) +
                     tf.reduce_mean(tf.square(self.f_pred)))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'gtol': 1e-8,
                                                                         'eps': 1e-8,
                                                                         'ftol': 1e-15})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    # --------------------------------------------------------------------------------------------
    # tl;dr Instead of just a simple random initialisation, we seem to be using
    # a more specific (random) initialisation for deep multi-layer NN that is 
    # more appropriate, as proposed by Glorot (xavier_init)
    #
    # See also Xavier Glorot et al. in
    # 'Understanding the difficulty of training deep feedforward neural networks'
    # see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    # --------------------------------------------------------------------------------------------

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
        #return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32) # Deprecated
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lower_bounds) / (self.upper_bounds - self.lower_bounds) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # --------------------------------------------------------------------------------------------
    # Run a NN model on a data to predict solution u, and differentiate it to get gradients as in 2020 Misyris et al. (4)
    # --------------------------------------------------------------------------------------------

    def pyhsics_net(self, time, power, delta_0, omega_0): # Love it when a typo in the beginning just propagates further
        u = self.neural_net(X=tf.concat(values=[time, power, delta_0, omega_0],
                                        axis=1),
                            weights=self.weights,
                            biases=self.biases)

        u_t = tf.gradients(u, time)[0]
        u_tt = tf.gradients(u_t, time)[0]
        f = self.lambda_m * u_tt + self.lambda_d * u_t + self.lambda_b * tf.math.sin(u) - power

        return u, u_t, f

    # --------------------------------------------------------------------------------------------
    # Add a function to print the currently calculated loss during the training procedure
    # --------------------------------------------------------------------------------------------

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):

        tf_dict = {self.time_u_tf: self.time_u,
                   self.power_u_tf: self.power_u,
                   self.delta_0_u_tf: self.delta_0_u,
                   self.omega_0_u_tf: self.omega_0_u,
                   self.time_f_tf: self.time_f,
                   self.power_f_tf: self.power_f,
                   self.delta_0_f_tf: self.delta_0_f,
                   self.omega_0_f_tf: self.omega_0_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    # --------------------------------------------------------------------------------------------
    # Run the saved sess operations (run the NN on given input data I assume), giving predictions on a dataset
    # --------------------------------------------------------------------------------------------

    def predict(self, X_predict):

        u_predict, u_t_predict = self.sess.run([self.u_pred, self.u_t_pred],
                                               feed_dict={self.time_u_tf: X_predict[:, 0:1],
                                                          self.power_u_tf: X_predict[:, 1:2],
                                                          self.delta_0_u_tf: X_predict[:, 2:3],
                                                          self.omega_0_u_tf: X_predict[:, 3:4]})

        return u_predict, u_t_predict


# --------------------------------------------------------------------------------------------
# Define an ODE solver of the swing equation
# --------------------------------------------------------------------------------------------
def solve_ode(t, y_initial, power, lambda_m, lambda_d, lambda_b):

    def vdp1(t, y):
        return np.array([y[1], 1/lambda_m * (power - lambda_d*y[1] - lambda_b * np.sin(y[0]))])

    y = np.zeros((len(t), len(y_initial)))   # array for solution
    y[0, :] = y_initial

    r = integrate.ode(vdp1).set_integrator('dopri5')  # choice of method
    r.set_initial_value(y_initial, t[0])   # initial values
    for ii in range(1, t.size):
        y[ii, :] = r.integrate(t[ii])   # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")

    return y


if __name__ == "__main__":
    
    
    # --------------------------------------------------------------------------------------------
    # Define the swing equation parameters:
    # Inertia m
    # Damping d
    # Connectivity b (composed of B_ij * V_i * V_j)
    # SMIB Swing equation: m*delta'' + d*delta' + b*delta - p = 0
    # --------------------------------------------------------------------------------------------
    lambda_m = 0.4
    lambda_d = 0.15
    lambda_b = 0.2

    # --------------------------------------------------------------------------------------------
    # Initialise number of nodes in the layers
    # Input layer: 4 inputs (time, power, delta, delta')
    # 2 hidden layers of 100 nodes each
    # Output layer
    # --------------------------------------------------------------------------------------------
    layers = [4, 100, 100, 1]

    # --------------------------------------------------------------------------------------------
    # COMMENT
    # What is Nf? Is Nf=100 appropriate for a sufficient behavior? If not, what is an appropriate value?
    # N_u is the number of training data points
    # N_f is the number of collocation points - used in the collocation method of numerically 
    # solving the ODE, correspond to the a number of points in the solution domain
    # Raissi et al. use up to N_f=10000... N_f=0 is equivalent of not using physics (just black box NN,
    # ignoring the equation), two orders of magnitude higher seems to give lowest errors (with higher N_u)
    # --------------------------------------------------------------------------------------------
    N_u = 10
    N_f = 100

    # --------------------------------------------------------------------------------------------
    # Initialise lower and upper bounds on the time through which the system evolves,
    # expected generator power, voltage angle (delta) and its derivative (angular frequency omega) 
    # --------------------------------------------------------------------------------------------
    bounds_time = [0.0, 10.0]
    bounds_power = [0.0, 0.4]
    bounds_delta_0 = [-np.pi, np.pi]
    bounds_omega_0 = [-0.5, 0.5]

    # --------------------------------------------------------------------------------------------
    # Create lower and upper bound vectors based on the system variable bounds above
    # --------------------------------------------------------------------------------------------
    lower_bounds = np.array([bounds_time[0],
                             bounds_power[0],
                             bounds_delta_0[0],
                             bounds_omega_0[0]])

    upper_bounds = np.array([bounds_time[1],
                             bounds_power[1],
                             bounds_delta_0[1],
                             bounds_omega_0[1]])

    # --------------------------------------------------------------------------------------------
    # We generate training system setpoints X_u, with variables t, p, delta, delta',
    # based on pyDOEs Latin-Hypercube implementation lhs
    # lhs creates a quasi-random sampling distribution of N_u points, for a dimension space of 3 in
    # this scenario (power, delta and delta'), ranging from 0 to 1, which we scale by the dimension's 
    # lower and upper bounds - these are initial system setpoints at t=0 for N_u cases
    # --------------------------------------------------------------------------------------------
    X_u = lower_bounds[1:] + (upper_bounds[1:] - lower_bounds[1:]) * lhs(3, N_u)
    X_u = np.concatenate([np.zeros((N_u, 1)), X_u], axis=1)

    # --------------------------------------------------------------------------------------------
    # Create a similar set point set to the one above, but defining it as
    # to be the converged solutions at the elapsed time 
    # --------------------------------------------------------------------------------------------
    X_f_train = lower_bounds + (upper_bounds - lower_bounds) * lhs(4, N_f)

    # --------------------------------------------------------------------------------------------
    # Initialise the PINN class using the above-defined system parameters and inputs
    # --------------------------------------------------------------------------------------------
    model = PhysicsInformedNN(X_u, X_f_train, layers, lower_bounds, upper_bounds, lambda_m, lambda_d, lambda_b)

    # --------------------------------------------------------------------------------------------
    # Train the model, timing it and printing the elapsed time
    # --------------------------------------------------------------------------------------------
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------
    # N_u = 10, N_f =    1,   t =    8s, mean errors: u = 0.26e01, u_t = 0.29e01
    # N_u = 10, N_f =  100,   t =  167s, mean errors: u = 0.67e00, u_t = 0.12e01
    # N_u = 10, N_f =  500,   t =  777s, mean errors: u = 0.81e00, u_t = 0.12e01 <- not running more, managed a lunch in this time lol
    # ^- should increase N_u as well, if we wanted lower errors,
    # results look like in Raissi et al. Table 1 N_u/N_f error
    #
    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------
    # Set up the ODE time
    # --------------------------------------------------------------------------------------------
    N_test = 10
    N_time_steps = 101
    time = np.linspace(bounds_time[0], bounds_time[1], N_time_steps)

    # --------------------------------------------------------------------------------------------
    # Create a random generator set points, similar to X_u for the NN
    # --------------------------------------------------------------------------------------------
    X_trajectory = lower_bounds[1:] + (upper_bounds[1:] - lower_bounds[1:]) * lhs(3, N_test)

    # --------------------------------------------------------------------------------------------
    # Initialise vectors to store the exact ODE solutions, along with the NN predictions and its errors
    # --------------------------------------------------------------------------------------------
    exact_solution_u = np.zeros((N_test, N_time_steps))
    exact_solution_u_t = np.zeros((N_test, N_time_steps))
    predicted_solution_u = np.zeros((N_test, N_time_steps))
    predicted_solution_u_t = np.zeros((N_test, N_time_steps))
    error = np.zeros((N_test, 2))

    # --------------------------------------------------------------------------------------------
    # Try the ODE vs NN model on a case N_test times
    # --------------------------------------------------------------------------------------------

    for ii in range(0, N_test):
        # --------------------------------------------------------------------------------------------
        # Finds the exact solution for the given initial set point
        # --------------------------------------------------------------------------------------------
        exact_solution = solve_ode(t=time,
                                   y_initial=[X_trajectory[ii, 1], X_trajectory[ii, 2]],
                                   power=X_trajectory[ii, 0],
                                   lambda_m=lambda_m,
                                   lambda_d=lambda_d,
                                   lambda_b=lambda_b)

        exact_solution_u[ii, :] = exact_solution[:, 0]
        exact_solution_u_t[ii, :] = exact_solution[:, 1]

        # --------------------------------------------------------------------------------------------
        # Take the same initial set point from X_trajectory to apply the trained NN model to
        # --------------------------------------------------------------------------------------------
        X_predict = np.concatenate([time.reshape((-1, 1)),
                                    np.repeat(X_trajectory[ii:ii + 1, :], N_time_steps, axis=0)],
                                   axis=1)

        # --------------------------------------------------------------------------------------------
        # Run the trained NN model on the set point
        # --------------------------------------------------------------------------------------------
        prediction = model.predict(X_predict)
        predicted_solution_u[ii, :] = prediction[0].reshape((1, -1))
        predicted_solution_u_t[ii, :] = prediction[1].reshape((1, -1))

        # --------------------------------------------------------------------------------------------
        # Give the normalised error between the ODE and the predicted NN solution in terms of a 2-norm (largest singular value)
        # --------------------------------------------------------------------------------------------
        error_u = np.linalg.norm(exact_solution_u[ii, :] - predicted_solution_u[ii, :], 2) / np.linalg.norm(
            exact_solution_u[ii, :], 2)
        error_u_t = np.linalg.norm(exact_solution_u_t[ii, :] - predicted_solution_u_t[ii, :], 2) / np.linalg.norm(
            exact_solution_u_t[ii, :], 2)
        error[ii, :] = np.array([error_u, error_u_t])
        print('Error u: %e, error u_t: %e' % (error_u, error_u_t))

    # --------------------------------------------------------------------------------------------
    # Plot NN predictions against the ODE solutions for delta and its derivative
    # --------------------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    for ii in range(0, N_test):
        ax1.plot(time, predicted_solution_u[ii, :], linewidth=0.8, color='C%i' % ii)
        ax1.plot(time, exact_solution_u[ii, :], linestyle='--', linewidth=0.8, color='C%i' % ii)

        ax2.plot(time, predicted_solution_u_t[ii, :], linewidth=0.8, color='C%i' % ii)
        ax2.plot(time, exact_solution_u_t[ii, :], linestyle='--', linewidth=0.8, color='C%i' % ii)

    ax1.set_ylabel('delta [rad]')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('omega [rad/s]')
    ax1.grid(True)
    ax2.grid(True)
    plt.show()

    # --------------------------------------------------------------------------------------------
    # Plot NN predictions against the ODE solutions for both delta and delta' 
    # --------------------------------------------------------------------------------------------
    for ii in range(0, N_test):
        plt.plot(predicted_solution_u[ii, :], predicted_solution_u_t[ii, :], linewidth=0.8, color='C%i' % ii)
        plt.plot(exact_solution_u[ii, :], exact_solution_u_t[ii, :], linestyle='--', linewidth=0.8, color='C%i' % ii)

    plt.xlabel('delta [rad]')
    plt.ylabel('omega [rad/s]')
    plt.grid(True)
    plt.show()
    
# =============================================================================
#     Not going to lie, the predictions look terrible to me when plotted
# =============================================================================
    
