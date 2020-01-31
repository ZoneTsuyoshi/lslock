"""
=====================================================
Inference with Locally and Spatially Uniform LOCK
=====================================================
This module implements the locally and spatially uniform LOCK
for Linear-Gaussian state space models
"""
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger("lslock")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import os
import math
import time
import multiprocessing as mp
import itertools

import numpy as np

from utils import array1d, array2d
from util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality



def _elementwise_local_calculation(i, j, A1, A2, y, P, Pmax, update_interval):
    local_A1 = A1[i] | A1[j]
    local_A2 = A2[i] | A2[j]
    g2l1 = np.where(local_A1)[0]
    g2l2 = np.where(local_A2)[0]
    ldim1 = len(g2l1)
    # ldim2 = len(g2l2)
    y1 = y[:-1, g2l2] # tau x ldim2
    y2 = y[1:, g2l1] # tau x ldim1
    P_local = P[np.ix_(g2l1, g2l2)] # ldim1 x ldim2

    Xi = np.zeros((ldim1*update_interval, Pmax)) # ldim1 tau x Pmax
    for a in range(Pmax):
        y_repeat = np.repeat(y1, ldim1, axis=0) # ldi1 tau x ldim2
        P_tile = np.tile(np.where(P_local==a+1, True, False), (update_interval, 1))
        y_repeat[~P_tile] = 0
        Xi[:,a] = y_repeat.sum(axis=1)

    theta_local = np.linalg.pinv(Xi) @ y2.reshape(-1)
    return theta_local[int(P[i,j] - 1)]



def _gridwise_local_calculation(i, A1, A2, y, P, Pmax, update_interval):
    g2l1 = np.where(A1[i])[0]
    g2l2 = np.where(A2[i])[0]
    ldim1 = len(g2l1)
    # ldim2 = len(g2l2)
    y1 = y[:-1, g2l2] # tau x ldim2
    y2 = y[1:, g2l1] # tau x ldim1
    P_local = P[np.ix_(g2l1, g2l2)] # ldim1 x ldim2

    Xi = np.zeros((ldim1*update_interval, Pmax)) # ldim1 tau x Pmax
    for a in range(Pmax):
        y_repeat = np.repeat(y1, ldim1, axis=0) # ldi1 tau x ldim2
        P_tile = np.tile(np.where(P_local==a+1, True, False), (update_interval, 1))
        y_repeat[~P_tile] = 0
        Xi[:,a] = y_repeat.sum(axis=1)

    theta_local = np.linalg.pinv(Xi) @ y2.reshape(-1)
    return theta_local[P[i][g2l1] - 1]



class LSLOCK(object) :
    """Implements the Locally and Spatially Uniform LOCK.
    This class implements the LSLOCK,
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The LSLOCK is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})` and :math:`F` in real-time. 
    As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt`.

    Args:
        observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
        transition_matrix [n_dim_sys, n_dim_sys] 
            or [n_dim_sys, n_dim_sys]{numpy-array, float}
            also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
        observation_matrix [n_dim_sys, n_dim_obs] {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
        transition_covariance [n_time-1, n_dim_sys, n_dim_sys]
             or [n_dim_sys, n_dim_sys]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance
        parameter_matrix [n_dim_obs, n_dim_obs] {numpy-array, float}
            also known as :math:`A`. matrix which indicates parameter number,
            same number mean same parameter. each element of the matrix corresponds to
            observation transition matrix, which controls transition from y_{t-1} to y_{t}.
        method {string}
            : method for localized calculation
            "elementwise": each element of transition matrix is calculated independently
            "gridwise": each column vector of transition matrix is calculated independently
        update_interval {int}
            interval of update transition matrix F
        eta (in (0,1))
            update rate for transition matrix F
        cutoff
            cutoff distance for update transition matrix F
        save_dir {str, directory-like}
            directory for saving transition matrices and filtered states.
            if this variable is `None`, cannot save them.
        advance_mode {bool}
            if True, calculate transition matrix before filtering.
            if False, calculate the matrix after filtering.
        n_dim_sys {int}
            dimension of system variable
        n_dim_obs {int}
            dimension of observation variable
        dtype {type}
            data type of numpy-array
        use_gpu {bool}
            wheather use gpu and cupy.
            if True, you need install package `cupy`.
            if False, set `numpy` for calculation.
        num_cpu {int} or `all`
            number of cpus duaring calculating transition matrix.
            you can set `all` or positive integer.


    Attributes:
        y : `observation`
        F : `transition_matrix`
        Q : `transition_covariance`
        H : `observation_matrix`
        R : `observation_covariance`
    """

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_matrix = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                parameter_matrix = None, method = "gridwise",
                update_interval = 1, eta = 0.9, cutoff = 1.0, 
                save_dir = None,
                advance_mode = False,
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = True, num_cpu = "all"):
        """Setup initial parameters.
        """
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrix, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_covariance, array2d, -2),
             (parameter_matrix, array2d, -2)],
            n_dim_obs
        )

        # self.y = _parse_observations(observation)
        self.y = self.xp.asarray(observation).copy()

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = self.xp.asarray(initial_mean, dtype = dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrix is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = self.xp.asarray(transition_matrix, dtype = dtype)

        if transition_covariance is not None:
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)
        else:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)

        if observation_matrix is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = self.xp.asarray(observation_matrix, dtype = dtype)
        self.HI = self.xp.linalg.pinv(self.H)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = self.xp.asarray(observation_covariance, dtype = dtype)

        if parameter_matrix is None:
            self.P = np.eye(self.n_dim_obs, dtype=dtype)
        else:
            if self.use_gpu:
                self.P = parameter_matrix.get()
            else:
                self.P = np.asarray(parameter_matrix, dtype = int)

        self.Pmax = int(self.P.max())
        self.A1 = self.P.astype(bool)
        self.A2 = self.A1.copy()
        for i in range(self.n_dim_obs):
            self.A2[i, np.any(self.A1[self.A1[i]], axis=0)] = True
        # self.A = self.P.astype(bool)

        if method in ["elementwise", "gridwise"]:
            self.method = method
        else:
            raise ValueError("Variable \"method\" only allows \"elementwise\" or \"gridwise\""
                + ". So, your setting \"{}\" need to be changed.".format(method))

        self.update_interval = int(update_interval)
        self.tm_count = 1

        if save_dir is None:
            self.save_change = False
        else:
            self.save_change = True
            self.save_dir = save_dir
            self.fillnum = len(str(int(self.y.shape[0] / self.update_interval)))
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(0).zfill(self.fillnum) + ".npy"), self.F)

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
        else:
            self.num_cpu = num_cpu
        print("Set number of cpus are {}.".format(self.num_cpu))

        self.advance_mode = advance_mode
        self.eta = eta
        self.cutoff = cutoff
        self.dtype = dtype
        self.times = self.xp.zeros(6)


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
            V_pred [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
            V_filt [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T):
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred = self.initial_covariance.copy()
            else:
                if self.advance_mode and t-1<T-self.update_interval and (t-1)%self.update_interval==0:
                    self._update_transition_matrix(t+self.update_interval-1)
                start_time = time.time()
                self._predict_update(t)
                self.times[0] = time.time() - start_time
            
            if self.xp.any(self.xp.isnan(self.y[t])):
                self.x_filt[t] = self.x_pred[t]
                self.V_filt = self.V_pred
            else :
                start_time = time.time()
                self._filter_update(t)
                self.times[1] = time.time() - start_time
                if (not self.advance_mode) and t-1>self.update_interval and (t-1)%self.update_interval==0:
                    self._update_transition_matrix(t)


        if self.save_change:
            self.xp.save(os.path.join(self.save_dir, "states.npy"), self.x_filt)


    def _predict_update(self, t):
        """Calculate fileter update

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        Q = _last_dims(self.Q, t - 1, 2)

        # calculate predicted distribution for time t
        self.x_pred[t] = self.F @ self.x_filt[t-1]
        self.V_pred = self.F @ self.V_filt @ self.F.T + Q


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t
        """
        # extract parameters for time t
        R = _last_dims(self.R, t, 2)

        # calculate filter step
        K = self.V_pred @ (
            self.H.T @ self.xp.linalg.inv(self.H @ (self.V_pred @ self.H.T) + R)
            )
        self.x_filt[t] = self.x_pred[t] + K @ (
            self.y[t] - (self.H @ self.x_pred[t])
            )
        self.V_filt = self.V_pred - K @ (self.H @ self.V_pred)


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        G = self.xp.zeros((self.n_dim_obs, self.n_dim_obs), dtype=self.dtype)

        start_time = time.time()

        if self.use_gpu:
            y = self.y[t-self.update_interval:t+1].get()
        else:
            y = self.y[t-self.update_interval:t+1]
        
        if self.method=="elementwise": # elementwise
            where_is_A = np.where(self.A1)
            # where_is_A = np.where(self.A)

            start_time2 = time.time()

            p = mp.Pool(self.num_cpu)
            G_local = p.starmap(_elementwise_local_calculation, zip(where_is_A[0],
                                                        where_is_A[1],
                                                        itertools.repeat(self.A1),
                                                        itertools.repeat(self.A2),
                                                        itertools.repeat(y),
                                                        itertools.repeat(self.P),
                                                        itertools.repeat(self.Pmax),
                                                        itertools.repeat(self.update_interval)))
            p.close()
            self.times[5] = time.time() - start_time2
            # G_local = np.zeros(len(where_is_A[0]))
            # for (k, i, j) in zip(range(len(where_is_A[0])), where_is_A[0], where_is_A[1]):
            #     G_local[k] = _local_calculation(i, j, self.A1, self.A2, y, self.P, self.Pmax, self.update_interval)
                # G_local[k] = _local_calculation(i, j, self.A, y, self.P, self.Pmax, self.update_interval)
            # G[self.A] = G_local
            G[self.A1] = G_local
        elif self.method=="gridwise":
            p = mp.Pool(self.num_cpu)
            G_local = p.starmap(_gridwise_local_calculation, zip(range(self.n_dim_obs),
                                                        itertools.repeat(self.A1),
                                                        itertools.repeat(self.A2),
                                                        itertools.repeat(y),
                                                        itertools.repeat(self.P),
                                                        itertools.repeat(self.Pmax),
                                                        itertools.repeat(self.update_interval)))
            p.close()
            G[self.A1] = list(itertools.chain.from_iterable(G_local))

            # for i in range(self.n_dim_obs):
            #     G[i][self.A1[i]] = _gridwise_local_calculation(i, self.A1, self.A2, y, self.P, self.Pmax, self.update_interval)

        self.times[2] += time.time() - start_time
        if self.tm_count==1:
            self.F = self.HI @ G @ self.H
        else:
            Fh = self.HI @ G @ self.H
            self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)
        self.times[3] += time.time() - start_time
        self.times[4] += 1

        if self.save_change:
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(self.tm_count).zfill(self.fillnum) + ".npy"), self.F)
        self.tm_count += 1


    def get_predicted_value(self, dim = None):
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_pred[0]
        except :
            self.forward()

        if dim is None:
            return self.x_pred
        elif dim <= self.x_pred.shape[1]:
            return self.x_pred[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_pred.shape[1] + '.')


    def get_filtered_value(self, dim = None):
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_filt[0]
        except :
            self.forward()

        if dim is None:
            return self.x_filt
        elif dim <= self.x_filt.shape[1]:
            return self.x_filt[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_filt.shape[1] + '.')