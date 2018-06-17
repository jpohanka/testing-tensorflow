import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.contrib import distributions as tfcd
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.distributions.bijectors


class StudentTCDF(tfb.Bijector):

    def __init__(self):
        self.studentT_dist = tfcd.StudentT(df=4, loc=0., scale=1.)
        super(StudentTCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="StudentTCDF")

    def _forward(self, y):
        # Inverse CDF of normal distribution.
        return self.studentT_dist.quantile(y)

    def _inverse(self, x):
        # CDF of normal distribution.
        return self.studentT_dist.cdf(x)

    def _inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.studentT_dist.log_prob(x)
