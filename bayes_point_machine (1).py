import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.svm import SVC

np.random.seed(0)

# Noise scale in the probit likelihood
eps = 0.3


# Dataset (with bias feature)
# Each point is augmented with a third coordinate = 1.
# This allows us to write the classifier as w^T x and
# include a bias term (intercept) inside w.
X = np.array([
    [1,0,1],
    [0,1,1],
    [1,1,1]
],dtype=float)

# Labels
y = np.array([1,1,-1])

n,d = X.shape


# Importance sampling to approximate the true Bayes point

# The Bayes classifier uses the posterior mean
#      E[w | D]
# under the model:
#      p(w) = N(0,I)
#      p(y_i | x_i, w) = Φ(y_i w^T x_i / ε)
#
# We approximate the posterior expectation via importance sampling.

def bayes_point_importance(X,y,N=400000):

    # Sample weights from the prior p(w)
    w_samples = np.random.randn(N,d)

    # Log weights for importance sampling
    logw = np.zeros(N)

    # Compute likelihood contribution of each data point
    for i in range(n):

        # z = margin scaled by epsilon
        z = y[i]*(w_samples @ X[i])/eps

        # Add log likelihood term log Φ(z)
        logw += np.log(norm.cdf(z)+1e-12)

    # Numerical stabilization
    logw -= np.max(logw)

    # Convert to normalized importance weights
    weights = np.exp(logw)
    weights /= np.sum(weights)

    # Posterior mean estimate E[w|D]
    return weights @ w_samples


# True Bayes point used as reference solution
w_true = bayes_point_importance(X,y)


# Bayes classifier vs SVM
# Train a linear SVM on the 2D data
svm = SVC(kernel="linear",C=1e6)
svm.fit(X[:,:2],y)

# Extract SVM weights and intercept
w_svm = svm.coef_[0]
b_svm = svm.intercept_[0]

# x-grid for plotting separating lines
xs = np.linspace(-1.5,2,200)

# SVM decision boundary: w1 x + w2 y + b = 0
ys_svm = -(w_svm[0]*xs + b_svm)/w_svm[1]

# Bayes decision boundary: w_true^T x = 0
ys_bayes = -(w_true[0]*xs + w_true[2])/w_true[1]

plt.figure(figsize=(5,5))

# Plot training points
for i in range(n):
    if y[i]==1:
        plt.scatter(X[i,0],X[i,1],marker='o',s=120, color = 'red')
    else:
        plt.scatter(X[i,0],X[i,1],marker='x',s=120)

# Plot separating lines
plt.plot(xs,ys_svm,label="SVM")
plt.plot(xs,ys_bayes,label="Bayes")

plt.title("Figure 3(a): Bayes vs SVM")
plt.legend()
plt.xlim(-1.5,2)
plt.ylim(-1.5,2)
plt.show()


# Error metric

# Measure distance between two separators.
# We normalize weights so that we compare directions,
# since scaling w does not change the decision boundary.
def error(w):

    w1 = w / np.linalg.norm(w)
    w2 = w_true / np.linalg.norm(w_true)

    return np.linalg.norm(w1 - w2)


# EP / ADF approximation

# Iteratively estimate the Bayes point using updates
# inspired by expectation propagation / adf.

def ep_algorithm(iterations=30):

    w = np.zeros(d)

    cost=[]
    err=[]

    prev_w=None

    for t in range(iterations):

        for i in range(n):

            # Probit margin
            z = y[i]*(X[i]@w)/eps

            # Moment-matching term φ(z)/Φ(z)
            g = norm.pdf(z)/(norm.cdf(z)+1e-12)

            # Update estimate of w
            w += 0.1*g*y[i]*X[i]

        # Record computational cost and error
        cost.append((t+1)*n*d)
        err.append(error(w))

        # Stop if algorithm converged
        if prev_w is not None and np.linalg.norm(w-prev_w) < 1e-6:
            break

        prev_w = w.copy()

    return np.array(cost),np.array(err)


# Mean-field approximation

# Similar iterative update but typically slower convergence.
def mean_field(iterations=30):

    w=np.zeros(d)

    cost=[]
    err=[]

    prev_w=None

    for t in range(iterations):

        for i in range(n):

            z=y[i]*(X[i]@w)/eps
            g=norm.pdf(z)/(norm.cdf(z)+1e-12)

            w+=0.05*g*y[i]*X[i]

        cost.append((t+1)*n*d)
        err.append(error(w))

        if prev_w is not None and np.linalg.norm(w-prev_w) < 1e-6:
            break

        prev_w=w.copy()

    return np.array(cost),np.array(err)


# TAP approximation

# TAP adds a second-order correction term to mean-field updates.

def tap_algorithm(iterations=30):

    w=np.zeros(d)

    cost=[]
    err=[]

    prev_w=None

    for t in range(iterations):

        for i in range(n):

            z=y[i]*(X[i]@w)/eps

            g=norm.pdf(z)/(norm.cdf(z)+1e-12)
            tap=g*(1-g*z)

            w+=0.05*tap*y[i]*X[i]

        cost.append((t+1)*n*d)
        err.append(error(w))

        if prev_w is not None and np.linalg.norm(w-prev_w) < 1e-6:
            break

        prev_w=w.copy()

    return np.array(cost),np.array(err)


# Billiard sampler (MCMC-style posterior sampling)

# Generates samples approximately from the posterior p(w|D)
# using a simple random-walk accept/reject procedure.

def billiard_sampler(steps=30000):

    w=np.zeros(d)

    samples=[]

    cost=[]
    err=[]

    for t in range(steps):

        # Random walk proposal
        w+=0.02*np.random.randn(d)

        accept=True

        # Accept with probability based on likelihood
        for i in range(n):
            if np.random.rand()>norm.cdf(y[i]*(X[i]@w)/eps):
                accept=False
                break

        if accept:
            samples.append(w.copy())

        # Once enough samples are collected, estimate posterior mean
        if len(samples)>50:

            m=np.mean(samples,axis=0)

            cost.append(t*d)
            err.append(error(m))

    return np.array(cost),np.array(err)


# Run all algorithms

ep_cost,ep_err = ep_algorithm()
mf_cost,mf_err = mean_field()
tap_cost,tap_err = tap_algorithm()
bill_cost,bill_err = billiard_sampler()

# Compute SVM error for comparison
svm_w = np.append(w_svm,b_svm)
svm_err = error(svm_w)


# Cost vs Error comparison

plt.figure(figsize=(6,5))

plt.loglog(ep_cost,ep_err,label="EP")
plt.loglog(mf_cost,mf_err,label="MF")
plt.loglog(tap_cost,tap_err,label="TAP")

# SVM plotted as a reference point
plt.scatter([1e5],[svm_err],label="SVM")

plt.xlabel("FLOPs")
plt.ylabel("Error")
plt.title("Figure 3(b): Cost vs Error")

plt.legend()

plt.show()