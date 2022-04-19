import os, sys
from getpass import getpass
import numpy as np
from math import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tfb = tfp.bijectors

import gpflow as gpf

from gpflow.config import default_float

class ZipLikelihood(gpf.likelihoods.MultiLatentTFPConditional):
    """
    Zero-inflated Poisson Likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    The `presence` and `rate` of the distribution are given by a
    two-dimensional multi-output GP.
    """

    def __init__(
        self,
         binsize=1.0,
        **kwargs,
    ):
        """
        :param binsize: array of areas for Poisson distribution
        """
        self.binsize = np.reshape(np.array(binsize, dtype=default_float()),(-1,1))

        def conditional_distribution(Fs) -> tfp.distributions.Distribution:
            tf.debugging.assert_equal(tf.shape(Fs)[-1], 2)
            f = (Fs[..., :1])
            g = (Fs[..., 1:])
            
            g = tf.math.sigmoid(g)
            h = tf.stack((1-g,g),axis=-1)
            
            return tfd.Mixture(cat=tfd.Categorical(probs=(h)),
                                      components=[tfd.Deterministic(loc=tf.zeros_like((f))),
                                      tfd.Poisson(log_rate= (f + tf.math.log(self.binsize)))])


        super().__init__(
            latent_dim=2, conditional_distribution=conditional_distribution, **kwargs,
        )


max_train=100
vals = np.arange(1,11)
Nr = 10


#def run_estimate(X_,wdata_):
    
    
    
for r in range(Nr):
    for v in vals:

        inputfile = 'simdata_p' + str(v) + '_r' + str(r) + '.csv'



        df =pd.read_csv('../data/simdata/' + inputfile)
        trueW = df[df.transect_id<0].wildebeest.values[0]
        df = df[df.transect_id>0]

        num_photo = len(df)

        X = df[['x','y']].values/1000
        wdata = df['wildebeest'].values[:,None].astype(np.float64)

        areas = df.photo_area.values/1000/1000


        data = (X, wdata)

class CovariateMeanFunction(gpf.mean_functions.MeanFunction):
    """
    y_i = A x_i + b
    """

    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be [D, Q], b must be a vector of length Q.
        """
        super().__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b
 
        # average intensity of images with wildebeest present
        mean_log_intensity = np.log(wdata[wdata>0].sum()/areas[wdata[:,0]>0].sum())

        # average probability of wildebeest present
        p = wdata[wdata>0].shape[0]/wdata.shape[0]
        p = min(0.99,p)
        mean_presence = np.log(p/(1.000-p))

        v1=5.
        v2=5.
        l1=1.
        l2=1.

        num_latent_gps=2


        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(X),  # This is U1 = f1(Z1)
                gpf.inducing_variables.InducingPoints(X),  # This is U2 = f2(Z2)
            ]
        )

        likelihood = ZipLikelihood(binsize=areas)


        kernel = gpf.kernels.SeparateIndependent(
            [
                gpf.kernels.Matern32(variance=v1,lengthscales=l1),  # This is k1, the kernel of f
                gpf.kernels.Matern32(variance=v2,lengthscales=l2),  # this is k2, the kernel of g

            ]
        )
        means = np.array([mean_log_intensity,mean_presence]).reshape((-1,1))
        mean_fn = gpf.mean_functions.Constant(c=means)


        svgp = gpf.models.SVGP(inducing_variable=inducing_variable,
                                kernel=kernel,
                                mean_function=mean_fn,
                                num_latent_gps=num_latent_gps,
                                likelihood = likelihood, q_diag=False  )
        gpf.set_trainable(svgp.inducing_variable.inducing_variable_list[0].Z,False)
        gpf.set_trainable(svgp.inducing_variable.inducing_variable_list[1].Z,False)

        f64 = np.float64
        svgp.kernel.kernels[0].lengthscales.prior = tfd.Gamma(f64(2.0), f64(0.5))
        svgp.kernel.kernels[0].variance.prior = tfd.Gamma(f64(2.0), f64(2.0))

        svgp.kernel.kernels[1].lengthscales.prior = tfd.Gamma(f64(2.0), f64(0.5))
        svgp.kernel.kernels[1].variance.prior = tfd.Gamma(f64(2.0), f64(2.0))


        opt = gpf.optimizers.Scipy()
        training_loss = svgp.training_loss_closure(data, compile=True)
        opt.minimize(training_loss, variables=svgp.trainable_variables,options=dict(maxiter=2000) )

        #for i in (range(0)):

        #    res = opt.minimize(training_loss, variables=svgp.trainable_variables,options=dict(maxiter=100, gtol=0.5))
        #    print(res)
        #    if res.success: break

        #v1 = svgp.kernel.kernels[0].variance.numpy()
        #v2 = svgp.kernel.kernels[1].variance.numpy()

        #l1 = svgp.kernel.kernels[0].lengthscales.numpy()
        #l2 = svgp.kernel.kernels[1].lengthscales.numpy()
        #means = svgp.mean_function.c.numpy()


        #kernel = gpf.kernels.SeparateIndependent(
        #    [
        #        gpf.kernels.Matern32(variance=v1,lengthscales=l1),  # This is k1, the kernel of f
        #        gpf.kernels.Matern32(variance=v2,lengthscales=l2),  # this is k2, the kernel of g

        #    ]
        #)
        #mean_fn = gpf.mean_functions.Constant(c=means)


        #svgp = gpf.models.SVGP(inducing_variable=inducing_variable,
        #                        kernel=kernel,
        #                        mean_function=mean_fn,
        #                        num_latent_gps=num_latent_gps,
        #                        likelihood = likelihood, q_diag=False  )

        #gpf.set_trainable(svgp.trainable_parameters,False)
        #gpf.set_trainable(svgp.q_mu, True)
        #gpf.set_trainable(svgp.q_sqrt, True)


        #opt = gpf.optimizers.Scipy()
        #training_loss = svgp.training_loss_closure(data, compile=True)
        #opt.minimize(training_loss, variables=svgp.trainable_variables)#,options=dict(maxiter=2000, gtol=0.05))

        ngrid=50
        pred_locations = np.array([np.linspace(0.5,49.5,ngrid),np.linspace(0.5,49.5,ngrid)])


        xx, yy = np.meshgrid(pred_locations[0], pred_locations[1], sparse=False)
        x_grid = np.array([xx.flatten(), yy.flatten()]).T

        nsamp = 20000
        y_pred = svgp.predict_f_samples(x_grid,nsamp).numpy()#+0.5*vgp.predict_f(x_grid)[1].numpy()
        f_pred = y_pred[...,0]
        g_pred = y_pred[...,1]



        samples = tfd.Poisson(log_rate=f_pred).sample()
        samples = tf.where(tfd.Bernoulli(probs=tf.nn.sigmoid(g_pred)).sample()>0,samples,tf.zeros_like(samples))
        totals = tf.reduce_sum(samples,axis=1).numpy()

        UCI = (np.percentile(totals,95))
        LCI = (np.percentile(totals,5))
        ml_est = (np.percentile(totals,50))
        print(inputfile, ": True value: ", trueW, " ML GP estimate: ", ml_est, ml_est/trueW, " (95% CI: ", LCI, LCI/trueW, ",",UCI, UCI/trueW,")", flush=True)

