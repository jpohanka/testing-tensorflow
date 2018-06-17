import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.distributions.bijectors

# class StudentTDCF(tfd.Bijector):
#
#     def __init__(self):


class NormalCDF(tfb.Bijector):

    """
    Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self):
        self.normal_dist = tfd.Normal(loc=0., scale=1.)
        super(NormalCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="NormalCDF")

    def _forward(self, y):
        # Inverse CDF of normal distribution.
        return self.normal_dist.quantile(y)

    def _inverse(self, x):
        # CDF of normal distribution.
        return self.normal_dist.cdf(x)

    def _inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.normal_dist.log_prob(x)


class GaussianCopulaTriL(tfd.TransformedDistribution):

    """Takes a location, and lower triangular matrix for the Cholesky factor."""

    def __init__(self, loc, scale_tril):
        super(GaussianCopulaTriL, self).__init__(
            distribution=tfd.MultivariateNormalTriL(
                loc=loc,
                scale_tril=scale_tril),
            bijector=tfb.Invert(NormalCDF()),
            validate_args=False,
            name="GaussianCopulaTriLUniform")


# Plot an example of this.
# unit_interval = np.linspace(0.01, 2.99, num=200, dtype=np.float32)
# x_grid, y_grid = np.meshgrid(unit_interval, unit_interval)
# coordinates = np.concatenate([x_grid[..., np.newaxis], y_grid[..., np.newaxis]], axis=-1)

# pdf = GaussianCopulaTriL(
#     loc=[0., 0.],
#     scale_tril=[[1., 0.2], [0.5, 0.6]],
# ).prob(coordinates)


from tensorflow.contrib.distributions.python.ops.gumbel import _Gumbel

a = 2.0
b = 2.0
gloc = 0.
gscale = 1.

x = tfd.Kumaraswamy(a, b)
y = _Gumbel(loc=gloc, scale=gscale)


class Concat(tfb.Bijector):
    """This bijector concatenates bijectors who act on scalars.

    More specifically, given [F_0, F_1, ... F_n] which are scalar transformations,
    this bijector creates a transformation which operates on the vector
    [x_0, ... x_n] with the transformation [F_0(x_0), F_1(x_1) ..., F_n(x_n)].


    NOTE: This class does no error checking, so use with caution.

    """

    def __init__(self, bijectors):
        self._bijectors = bijectors
        super(Concat, self).__init__(
            forward_min_event_ndims=1,
            validate_args=False,
            name="ConcatBijector")

    @property
    def bijectors(self):
        return self._bijectors

    def _forward(self, x):
        split_xs = tf.split(x, len(self.bijectors), -1)
        transformed_xs = [b_i.forward(x_i) for b_i, x_i in zip(
            self.bijectors, split_xs)]
        return tf.concat(transformed_xs, -1)

    def _inverse(self, y):
        split_ys = tf.split(y, len(self.bijectors), -1)
        transformed_ys = [b_i.inverse(y_i) for b_i, y_i in zip(
            self.bijectors, split_ys)]
        return tf.concat(transformed_ys, -1)

    def _forward_log_det_jacobian(self, x):
        split_xs = tf.split(x, len(self.bijectors), -1)
        fldjs = [
            b_i.forward_log_det_jacobian(x_i, event_ndims=0) for b_i, x_i in zip(
                self.bijectors, split_xs)]
        return tf.squeeze(sum(fldjs), axis=-1)

    def _inverse_log_det_jacobian(self, y):
        split_ys = tf.split(y, len(self.bijectors), -1)
        ildjs = [
            b_i.inverse_log_det_jacobian(y_i, event_ndims=0) for b_i, y_i in zip(
                self.bijectors, split_ys)]
        return tf.squeeze(sum(ildjs), axis=-1)


class WarpedGaussianCopula(tfd.TransformedDistribution):
    """Application of a Gaussian Copula on a list of target marginals.

    This implements an application of a Gaussian Copula. Given [x_0, ... x_n]
    which are distributed marginally (with CDF) [F_0, ... F_n],
    `GaussianCopula` represents an application of the Copula, such that the
    resulting multivariate distribution has the above specified marginals.

    The marginals are specified by `marginal_bijectors`: These are
    bijectors whose `inverse` encodes the CDF and `forward` the inverse CDF.
    """

    def __init__(self, loc, scale_tril, marginal_bijectors):
        super(WarpedGaussianCopula, self).__init__(
            distribution=GaussianCopulaTriL(loc=loc, scale_tril=scale_tril),
            bijector=Concat(marginal_bijectors),
            validate_args=False,
            name="GaussianCopula")

    @property
    def marginal_bijectors(self):
        """List of bijectors which correspond to marginals."""


# Create our coordinates:
x_axis_interval = np.linspace(0.01, 0.99, num=200, dtype=np.float32)
y_axis_interval = np.linspace(-2., 3., num=200, dtype=np.float32)
x_grid, y_grid = np.meshgrid(x_axis_interval, y_axis_interval)
coordinates = np.concatenate(
    [x_grid[..., np.newaxis], y_grid[..., np.newaxis]], -1)


def create_gaussian_copula(correlation):
    # Use Gaussian Copula to add dependence.
    return WarpedGaussianCopula(
        loc=[0.,  0.],
        scale_tril=[[1., 0.], [correlation, tf.sqrt(1. - correlation ** 2)]],
        # These encode the marginals we want. In this case we want X_0 has
        # Kumaraswamy marginal, and X_1 has Gumbel marginal.

        marginal_bijectors=[
            tfb.Kumaraswamy(a, b),
            # Kumaraswamy follows the above convention, while
            # Gumbel does not, and has to be inverted.
            tfb.Invert(tfb.Gumbel(loc=0., scale=1.))])


# Note that the zero case will correspond to independent marginals!
correlations = [0., -0.8, 0.8]
copulas = []
probs = []
for correlation in correlations:
    copula = create_gaussian_copula(correlation)
    copulas.append(copula)
    probs.append(copula.prob(coordinates))


# Plot it's density
with tf.Session() as sess:
    copula_evals = sess.run(probs)

for correlation, copula_eval in zip(correlations, copula_evals):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x_grid, y_grid, copula_eval, 100, cmap=plt.cm.jet)
    # plt.contour(x_grid, y_grid, copula_eval, 100, cmap=plt.cm.jet)
    plt.show()
