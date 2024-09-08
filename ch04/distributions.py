import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow_probability as tfp

tfd = tfp.distributions


def main():
    mu = 0
    sigma = 1.5
    gaussian_dist = tfd.Normal(loc=mu, scale=sigma)

    samples = gaussian_dist.sample(1000)
    sns.histplot(samples, stat="probability", kde=True)
    plt.show()

    pdf_range = np.arange(-4, 4, 0.1)
    pdf_values = []
    for x in pdf_range:
        pdf_values.append(gaussian_dist.prob(x))
    plt.figure(figsize=(10, 5))
    plt.plot(pdf_range, pdf_values)
    plt.title("Probability density function", fontsize="15")
    plt.xlabel("x", fontsize="15")
    plt.ylabel("probability", fontsize="15")
    plt.show()

    cdf_range = np.arange(-4, 4, 0.1)
    cdf_values = []
    for x in cdf_range:
        cdf_values.append(gaussian_dist.cdf(x))
    plt.figure(figsize=(10, 5))
    plt.plot(cdf_range, cdf_values)
    plt.title("Cumulative density function", fontsize="15")
    plt.xlabel("x", fontsize="15")
    plt.ylabel("CDF", fontsize="15")
    plt.show()

    mu = gaussian_dist.mean().numpy()
    sigma = gaussian_dist.stddev().numpy()
    print(f"{mu=}, {sigma=}")

    x = 5
    log_likelihood = gaussian_dist.log_prob(x)
    negative_log_likelihood = -log_likelihood.numpy()
    print(f"{negative_log_likelihood=}")


if __name__ == "__main__":
    main()
