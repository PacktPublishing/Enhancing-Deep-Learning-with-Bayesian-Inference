import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns

"""
This code generates the sampling figures used in Chapter 2.
"""

# Set seed to reproduce exact figures from the book
np.random.seed(55)


def generate_example_data():
    df = pd.DataFrame({"height (cm)": np.random.normal(loc=165, scale=30, size=(100000,))})
    df = df[df["height (cm)"] < 205]
    return df


def generate_random_sampling_plot(df: pd.DataFrame, n_samples: int = 100):
    # Generate distribution plot from random samples

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

    df_sample = df.sample(n_samples)

    sns.kdeplot(x="height (cm)", data=df, ax=axs[0], fill=True)
    axs[0].title.set_text("True distribution")
    sns.kdeplot(x="height (cm)", data=df_sample, ax=axs[1], fill=True)
    axs[1].title.set_text(f"Sample distribution: {n_samples} samples")
    plt.show()


def generate_metropolis_sampling_plot(df: pd.DataFrame, n_samples: int = 100):
    # Generate distribution plot from metropolis samples
    __spec__ = None  # noqa:F841

    df_sample = df.sample(n_samples)
    y = df_sample["height (cm)"]

    niter = n_samples
    with pm.Model() as _:
        # define priors
        mu = pm.Uniform("mu", lower=30, upper=180, shape=())
        sigma = pm.Uniform("sigma", lower=0, upper=30, shape=())

        # define likelihood
        pm.TruncatedNormal("Y_obs", mu=mu, sd=sigma, upper=205, observed=y)

        # inference
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(niter, step, start, chains=1)
        ppc = pm.sample_posterior_predictive(
            trace, var_names=["mu", "sigma", "Y_obs"], random_seed=55
        )

    mc_samples = []
    for i in range(niter):
        sample = trace[i]["mu"]
        mc_samples.append(sample)

    mc_samples = ppc["Y_obs"][:-1]

    mc_samples = np.array(mc_samples)
    df_mc_sample = pd.DataFrame({"height (cm)": mc_samples.reshape(-1)})

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    sns.kdeplot(x="height (cm)", data=df, ax=axs[0], fill=True)
    axs[0].title.set_text("True distribution")

    sns.kdeplot(x="height (cm)", data=df_mc_sample, ax=axs[1], fill=True)
    axs[1].title.set_text(f"Sample distribution using MCMC: {niter} samples")

    plt.show()


if __name__ == "__main__":
    df = generate_example_data()
    generate_random_sampling_plot(df, 10)
    generate_metropolis_sampling_plot(df, 10)
