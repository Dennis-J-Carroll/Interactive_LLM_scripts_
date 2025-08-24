
# -*- coding: utf-8 -*-
"""
Bayesian Statistics Interactive Lesson

This script provides an introduction to Bayesian statistics and modeling. We'll
explore the key concepts and build a simple Bayesian linear regression model.

Author: Gemini
Date: 2025-08-24
"""

# #############################################################################
# # Part 1: Introduction to Bayesian Statistics                            #
# #############################################################################

# Welcome to the world of Bayesian statistics! Unlike frequentist statistics,
# which focuses on the frequency of events, Bayesian statistics is about
# updating our beliefs in the light of new evidence.

# ---
# Key Bayesian Concepts You'll Encounter:
# ---
# 1.  **Prior Distribution:** This represents our initial beliefs about a
#     parameter before we see any data.
# 2.  **Likelihood:** This is the probability of observing the data given a
#     particular value of the parameter.
# 3.  **Posterior Distribution:** This is the updated distribution of our beliefs
#     about the parameter after we have seen the data. It is calculated by
#     combining the prior and the likelihood using Bayes' theorem.
# 4.  **Markov Chain Monte Carlo (MCMC):** This is a class of algorithms for
#     sampling from a probability distribution. We'll use it to approximate the
#     posterior distribution.

# #############################################################################
# # Part 2: Setup and Data Generation                                      #
# #############################################################################

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# ---
# Data Generation
# ---
# We'll generate some synthetic data for a simple linear regression problem.

np.random.seed(42)
alpha_true = 2.5
beta_true = 1.3
noise_true = 0.5

X = np.linspace(0, 10, 100)
y = alpha_true + beta_true * X + np.random.normal(0, noise_true, 100)

# #############################################################################
# # Part 3: Building the Bayesian Linear Regression Model                  #
# #############################################################################

with pm.Model() as linear_model:
    # ---
    # Priors
    # ---
    # We'll define our prior beliefs about the model parameters.
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    noise = pm.HalfNormal("noise", sigma=1)

    # ---
    # Likelihood
    # ---
    # This is where we connect the parameters to the data.
    mu = alpha + beta * X
    y_obs = pm.Normal("y_obs", mu=mu, sigma=noise, observed=y)

    # ---
    # MCMC Sampling
    # ---
    # We'll use MCMC to sample from the posterior distribution.
    trace = pm.sample(2000, tune=1000)

# #############################################################################
# # Part 4: Analyzing the Results                                          #
# #############################################################################

# ---
# Posterior Distributions
# ---
# Let's look at the posterior distributions of our model parameters.

pm.plot_posterior(trace)
plt.show()

# ---
# Expert Bayesian Statistician's Thought:
# ---
# "The posterior plots are the main output of a Bayesian analysis. They show us
# the range of plausible values for our parameters, given the data and our prior
# beliefs. We can see that the true values of our parameters (alpha=2.5,
# beta=1.3) are well within the high-density regions of the posterior
# distributions."

# ---
# Posterior Predictive Checks
# ---
# We can also use the posterior distribution to generate new data and see how
# well it matches the original data.

with linear_model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

plt.scatter(X, y, label="Observed data")
plt.plot(X, ppc.posterior_predictive["y_obs"].mean(axis=(0,1)), color="r", label="Posterior predictive mean")
plt.title("Posterior Predictive Check")
plt.legend()
plt.show()
