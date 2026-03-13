import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, rand
from scipy.stats import norm
from scipy.special import logsumexp

np.random.seed(0)

# Model parameters
w = 0.5
prior_var = 100.0
signal_var = 1.0
clutter_var = 10.0
true_x = 2.0

# Data generation
def generate_data(n):
    mask = rand(n) < (1 - w)
    y = np.zeros(n)
    y[mask] = true_x + np.sqrt(signal_var)*randn(np.sum(mask))
    y[~mask] = np.sqrt(clutter_var)*randn(np.sum(~mask))
    return y

# True posterior + evidence 
def true_posterior_and_logZ(y, grid):

    dx = grid[1] - grid[0]

    log_prior = norm.logpdf(grid, 0, np.sqrt(prior_var))

    log_lik = np.zeros_like(grid)
    for yi in y:
        log_lik += np.log(
            (1-w)*norm.pdf(yi, grid, np.sqrt(signal_var))
            + w*norm.pdf(yi, 0, np.sqrt(clutter_var))
        )

    log_joint = log_prior + log_lik

    logZ = logsumexp(log_joint) + np.log(dx)

    log_post = log_joint - logZ
    posterior = np.exp(log_post)

    return posterior, logZ


def adf(w, yi, m_cav, v_cav):

    Zs = (1-w)*norm.pdf(yi, m_cav, np.sqrt(v_cav+signal_var))
    Zc = w*norm.pdf(yi, 0, np.sqrt(clutter_var))
    Z = Zs + Zc

    r = Zs/Z

    m_tilt = m_cav + (v_cav*r*(yi-m_cav))/(v_cav+signal_var)

    v_tilt = (
        v_cav
        - (v_cav**2*r)/(v_cav+signal_var)
        + (v_cav**2*r*(1-r)*(yi-m_cav)**2)/(v_cav+signal_var)**2
    )

    return Z, m_tilt, v_tilt



# EP
def ep_curves(y, true_mean, true_logZ, max_iter=40, damping=0.7):

    # Number of observations (n in the paper)
    n = len(y)


    # Instead of coding with m_i and v_i, we do a variable change:
    # τ_i = 1/v_i
    # η_i = m_i/v_i
    # These parameters are more natural for a gaussian density
    # and simply the expressions
    # Each likelihood term is approximated by
    #     t_i(x) = s_i exp(η_i x - ½ τ_i x^2)
    #
    # In the paper this is written as a Gaussian
    # with parameters (m_i, v_i, s_i).
    #
    # Initially v_i = ∞  -> τ_i = 0
    # so the sites contribute nothing.

    tau_i = np.zeros(n)
    eta_i = np.zeros(n)
    s_i = np.ones(n)

    # Initialize posterior q(x)
    #
    # Initially q(x) = prior
    # prior: x ~ N(0, prior_var)
    #
    # In natural parameters:
    # τ = 1/prior_var
    # η = 0
    tau_post = 1/prior_var
    eta_post = 0.0

    mean_err = []
    ev_err = []
    cost = []


    for it in range(max_iter):

        for i in range(n):

            # Step 3(a): Remove site i
            # Compute the cavity distribution
            # q_{-i}(x) ∝ q(x) / t_i(x)
            # In natural parameters subtraction is easy:
            # τ_cav = τ_post − τ_i
            # η_cav = η_post − η_i
            tau_cav = tau_post - tau_i[i]
            eta_cav = eta_post - eta_i[i]

            # Convert cavity distribution to mean/variance form
            v_cav = 1/tau_cav
            m_cav = eta_cav/tau_cav

            # Step 3(b): Form the tilted distribution
            # We just call the adf function defined above

            Z, m_tilt, v_tilt = adf(w,y[i],m_cav,v_cav)
            

            # Step 3(c): Update the site approximation
            # We choose τ_i and η_i so that
            # q_new(x) = t_i(x) q_{-i}(x)
            # has mean m_tilt and variance v_tilt
            # Using natural parameters:
            # τ_new = 1/v_tilt − τ_cav
            # η_new = m_tilt/v_tilt − η_cav


            # perform check so that no nan
            if v_tilt <= 1e-12 or not np.isfinite(v_tilt):
                continue

            tau_new = 1/v_tilt - tau_cav
            eta_new = m_tilt/v_tilt - eta_cav

            # we use damping otherwise results are terrible
            tau_i[i] = (1-damping)*tau_i[i] + damping*tau_new
            eta_i[i] = (1-damping)*eta_i[i] + damping*eta_new

            # so that no warnings
            if tau_new <= 1e-12 or not np.isfinite(tau_new):
                continue

            v_site = 1 / tau_new
            m_site = eta_new / tau_new

            den = np.sqrt(2*np.pi*v_site) * norm.pdf(
                m_site,
                m_cav,
                np.sqrt(v_site + v_cav)
            )
            # so that no warnings
            if den <= 0 or not np.isfinite(den):
                continue

            s_i[i] = Z / den
            

            # Recompute posterior parameters
            # τ_post = τ_prior + Σ τ_i
            # η_post = Σ η_i
            # Because Gaussian natural parameters add.
            tau_post = 1/prior_var + np.sum(tau_i)
            eta_post = np.sum(eta_i)

        # Compute posterior mean and variance
        v_x = 1/tau_post
        m_x = eta_post/tau_post

        # Step 4: Approximate model evidence

        # compute site means and variances
        mask = tau_i > 1e-12
        v_i = 1 / tau_i[mask]
        m_i = eta_i[mask] / tau_i[mask]

        # compute B term from the paper
        B = m_x**2 / v_x - np.sum(m_i**2 / v_i)

        logZ_est = (
            0.5*np.log(2*np.pi*v_x)
            + 0.5*B
            + np.sum(np.log(s_i + 1e-12))
        )

        mean_err.append(abs(m_x - true_mean))
        ev_err.append(abs(logZ_est - true_logZ))
        cost.append((it+1)*n)


    return np.array(cost), np.array(mean_err), np.array(ev_err)



# Laplace
def laplace_curves(y, true_mean, true_logZ, max_iter=60):

    # Number of observations
    n = len(y)

    # Current estimate of the posterior mode (MAP estimate of x)
    # Laplace approximation expands the log posterior around this point
    x = 0.0

    mean_err = []   
    ev_err = []     
    cost = []       

    # Small fixed step size for gradient ascent
    # We use gradiant descent and not Newton step since the latter wasn't working well
    step_size = 0.01

    for it in range(max_iter):

        # Step 1: Compute log posterior and its gradient

        # Prior contribution to log posterior
        # Prior: x ~ N(0, prior_var)
        logp = -0.5*x**2/prior_var - 0.5*np.log(2*np.pi*prior_var)

        # Gradient of the log prior
        g = -x/prior_var

        # Add contributions from each likelihood term
        for yi in y:

            # Likelihood model:
            # p(y_i | x) =
            # (1-w) N(y_i | x, signal_var)
            # + w N(y_i | 0, clutter_var)
            # We compute this in log-space for stability

            log_a = np.log(1-w) + norm.logpdf(yi, x, np.sqrt(signal_var))
            log_b = np.log(w) + norm.logpdf(yi, 0, np.sqrt(clutter_var))

            # logaddexp implements
            # log(exp(log_a) + exp(log_b))
            log_denom = np.logaddexp(log_a, log_b)

            # Posterior probability that observation yi
            # came from the "signal" component
            r = np.exp(log_a - log_denom)

            # Gradient contribution from this observation
            # (derivative of log mixture likelihood)
            g += r * (yi - x) / signal_var

            # Add log likelihood contribution to log posterior
            logp += log_denom

        # Step 2: Update x using gradient ascent

        # Move x slightly in the direction of the gradient
        # This searches for the MAP estimate
        x = x + step_size * g

        # Step 3: Compute Hessian of log posterior at the mode

        # Second derivative of the log prior
        h = -1/prior_var

        for yi in y:

            # Recompute mixture quantities at the new x
            log_a = np.log(1-w) + norm.logpdf(yi, x, np.sqrt(signal_var))
            log_b = np.log(w) + norm.logpdf(yi, 0, np.sqrt(clutter_var))
            log_denom = np.logaddexp(log_a, log_b)

            r = np.exp(log_a - log_denom)

            # Second derivative of the log likelihood
            # for the mixture model
            h += -r/signal_var + r*(1-r)*(yi - x)**2 / signal_var**2

        # Step 4: Laplace variance approximation

        # Laplace approximates the posterior by
        # q(x) = N(x_MAP , -1/H)
        # where H is the Hessian of the log posterior
        if h < 0:
            var = -1/h
        else:
            # fallback if curvature becomes invalid
            print("Laplace method: invalid curvature")
            var = 1.0

        # Step 5: Laplace evidence approximation

        # Evidence approximation:
        #
        # log p(y) ≈ log p(y, x_MAP)
        #           + 1/2 log(2π var)
        logZ = logp + 0.5*np.log(2*np.pi*var)

        mean_err.append(abs(x - true_mean))
        ev_err.append(abs(logZ - true_logZ))
        cost.append((it+1)*n)

    return np.array(cost), np.array(mean_err), np.array(ev_err)



# VB
def vb_curves(y, true_mean, true_logZ, max_iter=40):

    # Number of observations
    n = len(y)

    # r_i will represent the variational probability that
    # observation y_i came from the "signal" Gaussian rather
    # than the clutter Gaussian.
    # Initialize them to 0.5 (complete uncertainty).
    r = np.ones(n)*0.5

    # Variational approximation for the posterior of x:
    # q(x) = N(m, v)
    # Start with prior parameters
    m = 0
    v = prior_var

    mean_err = []   
    ev_err = []     
    cost = []       

    for it in range(max_iter):

        # Step 1: Update variational responsibilities r_i
        # These represent the probability that each data
        # point belongs to the "signal" component.
        # This corresponds to the E-step in a mixture model.
        for i in range(n):

            # log probability that y_i came from signal
            log_s = np.log(1-w) + norm.logpdf(
                y[i], m, np.sqrt(v + signal_var)
            )

            # log probability that y_i came from clutter
            log_c = np.log(w) + norm.logpdf(
                y[i], 0, np.sqrt(clutter_var)
            )

            # Convert to probability using log-sum-exp
            # r_i = P(signal | y_i)
            r[i] = np.exp(log_s - np.logaddexp(log_s, log_c))


        # Step 2: Update the variational posterior q(x)
        # Under the mean-field approximation we assume
        # q(x) = N(m, v)
        # The update comes from matching natural parameters
        # of a Gaussian posterior.

        # Precision (inverse variance) of q(x)
        tau = 1/prior_var + np.sum(r)/signal_var

        # Natural mean parameter
        eta = np.sum(r*y)/signal_var

        # Convert natural parameters back to mean/variance
        v = 1/tau
        m = eta/tau


        # Step 3: Variational evidence estimate
        # Variational inference maximizes a lower bound
        # on the marginal likelihood (ELBO).
        # Here the simplified estimate is proportional to
        # the log determinant of the posterior precision.

        logZ_est = -0.5*np.log(tau)


        mean_err.append(abs(m - true_mean))
        ev_err.append(abs(logZ_est - true_logZ))
        cost.append((it+1)*n)

    return np.array(cost), np.array(mean_err), np.array(ev_err)

# Importance Sampling
def importance_curves(y, true_mean, true_logZ):

    # Number of observations
    n = len(y)

    mean_err = []   
    ev_err = []     
    cost = []       

    # Loop over different numbers of importance samples
    # (log-spaced so we test small and large sample sizes)
    for N in np.logspace(2,5,15).astype(int):

        # Step 1: Draw samples from the proposal distribution
        # The proposal distribution is the prior in this case:
        # x ~ N(0, prior_var)
        # This gives N candidate values of x.
        samples = np.sqrt(prior_var)*randn(N)

        # Step 2: Compute the log joint probability
        # log p(x, y)

        # Start with log prior
        log_joint = norm.logpdf(samples,0,np.sqrt(prior_var))

        # Add log likelihood contributions from each observation
        for yi in y:

            # Likelihood model:
            # p(y_i | x) =
            # (1-w) N(y_i | x, signal_var)
            # + w N(y_i | 0, clutter_var)
            #
            # This is a mixture of two Gaussians.
            log_joint += np.log(
                (1-w)*norm.pdf(yi,samples,np.sqrt(signal_var))
                + w*norm.pdf(yi,0,np.sqrt(clutter_var))
            )

        # Step 3: Estimate the marginal likelihood (evidence)
        # p(y) = ∫ p(x,y) dx
        # Importance sampling approximation:
        # p(y) ≈ (1/N) Σ p(x_k , y)
        # The logsumexp trick prevents numerical overflow.
        logZ_est = logsumexp(log_joint) - np.log(N)

        # Step 4: Compute normalized importance weights
        # w_k = p(x_k , y) / Σ p(x_j , y)
        # These weights approximate the posterior distribution.
        weights = np.exp(log_joint - logsumexp(log_joint))

        # Step 5: Estimate the posterior mean
        # E[x | y] ≈ Σ w_k x_k
        mean_est = np.sum(weights*samples)

        mean_err.append(abs(mean_est - true_mean))
        ev_err.append(abs(logZ_est - true_logZ))
        cost.append(N*n)

    return np.array(cost), np.array(mean_err), np.array(ev_err)

# Gibbs
def gibbs_curves(y, true_mean):

    # Initialize latent variable x (the unknown signal we want to infer)
    x = 0

    # z_i are latent indicators telling us whether each observation
    # comes from the signal component (z_i = 1) or clutter (z_i = 0)
    z = np.zeros(len(y))

    # Store posterior samples of x
    samples = []

    mean_err = []   
    cost = []       

    # Run Gibbs sampler
    for step in range(4000):

        # Step 1: Sample z_i (signal vs clutter indicators)
        # For each observation y_i we sample whether it
        # came from the signal or clutter component.
        for i in range(len(y)):

            # Probability of y_i under signal model
            p_s = (1-w)*norm.pdf(y[i],x,np.sqrt(signal_var))

            # Probability of y_i under clutter model
            p_c = w*norm.pdf(y[i],0,np.sqrt(clutter_var))

            # Sample z_i from Bernoulli distribution:
            # z_i = 1 (signal) with probability
            # p_s / (p_s + p_c)
            z[i] = rand() < p_s/(p_s+p_c)

        # Step 2: Sample x given z
        # Only observations marked as signal influence x
        active = y[z==1]

        # Compute posterior precision (inverse variance)
        tau = 1/prior_var + len(active)/signal_var

        # Natural mean parameter
        eta = np.sum(active)/signal_var

        # Sample x from its conditional posterior:
        # x | z,y ~ N(eta/tau , 1/tau)
        x = eta/tau + np.sqrt(1/tau)*randn()

        # Step 3: Store samples after burn-in
        # Skip the first 1000 iterations (burn-in)
        # and store every 50th sample to reduce correlation.
        if step>1000 and step%50==0:
            samples.append(x)
            mean_err.append(abs(np.mean(samples)-true_mean))
            cost.append(step*len(y))

    return np.array(cost), np.array(mean_err)

# Plot full 2x2 figure
def run_full():

    fig, axes = plt.subplots(2,2, figsize=(12,10))

    for col,n in enumerate([50,200]):

        y = generate_data(n)
        grid = np.linspace(-10,10,5000)
        true_post,true_logZ = true_posterior_and_logZ(y,grid)
        true_mean = np.sum(grid*true_post)*(grid[1]-grid[0])

        ep_c,ep_m,ep_e = ep_curves(y,true_mean,true_logZ)
        lap_c,lap_m,lap_e = laplace_curves(y,true_mean,true_logZ)
        vb_c,vb_m,vb_e = vb_curves(y,true_mean,true_logZ)
        is_c,is_m,is_e = importance_curves(y,true_mean,true_logZ)
        gib_c,gib_m = gibbs_curves(y,true_mean)

        ax = axes[0,col]
        ax.loglog(ep_c,ep_e,label="EP",marker='x')
        ax.loglog(lap_c,lap_e,label="Laplace")
        ax.loglog(vb_c,vb_e,label="VB")
        ax.loglog(is_c,is_e,label="Importance")
        ax.set_title(f"Evidence (n={n})")
        ax.set_xlabel("Cost")
        ax.set_ylabel("Error")
        ax.legend()

        ax = axes[1,col]
        ax.loglog(ep_c,ep_m,label="EP",marker='x',color='orange')
        ax.loglog(lap_c,lap_m,label="Laplace",color='red')
        ax.loglog(vb_c,vb_m,label="VB",color='green')
        ax.loglog(is_c,is_m,label="Importance", color='purple')
        ax.loglog(gib_c,gib_m,label="Gibbs", color='brown')
        ax.set_title(f"Posterior Mean (n={n})")
        ax.set_xlabel("Cost")
        ax.set_ylabel("Error")
        ax.legend()

    plt.tight_layout()
    plt.show()


#run_full()




def posterior_shape_experiment():

    np.random.seed(3)

    # dataset encouraging multimodality
    y = np.array([
        2.2, 2.0, 1.9, 2.3,
        -2.1, -2.0, -1.8, -2.2,
        0.2, -0.1, 0.1, -0.2,
        0.5, -0.5,
        1.0, -1.0,
        2.1, 2.0,
        -2.3, -2.1
    ])

    grid = np.linspace(-6,6,2000)
    dx = grid[1]-grid[0]

    # Exact posterior
    true_post, true_logZ = true_posterior_and_logZ(y,grid)
    true_mean = np.sum(grid*true_post)*dx

    ep_c,ep_m,_ = ep_curves(y,true_mean,true_logZ)
    lap_c,lap_m,_ = laplace_curves(y,true_mean,true_logZ)
    vb_c,vb_m,_ = vb_curves(y,true_mean,true_logZ)
    is_c,is_m,_ = importance_curves(y,true_mean,true_logZ)
    gib_c,gib_m = gibbs_curves(y,true_mean)

    # Use final mean estimates
    m_ep = true_mean - ep_m[-1]
    m_lap = true_mean - lap_m[-1]
    m_vb = true_mean - vb_m[-1]

    # Simple Gaussian approximations
    ep_curve = norm.pdf(grid,m_ep,1)
    lap_curve = norm.pdf(grid,m_lap,1)
    vb_curve = norm.pdf(grid,m_vb,1)

    plt.figure(figsize=(7,5))

    plt.plot(grid,true_post,label="Exact",linewidth=2)
    plt.plot(grid,ep_curve,label="EP")
    plt.plot(grid,vb_curve,label="VB")
    plt.plot(grid,lap_curve,label="Laplace")

    plt.xlabel("θ")
    plt.ylabel("Posterior density")
    plt.title("Posterior comparison")

    plt.legend()
    plt.xlim(-6,6)

    plt.show()

# Cost vs accuracy (posterior mean)

def figure2b():

    n = 20
    y = generate_data(n)

    grid = np.linspace(-10,10,5000)
    true_post, true_logZ = true_posterior_and_logZ(y,grid)

    dx = grid[1]-grid[0]
    true_mean = np.sum(grid * true_post) * dx

    ep_c,ep_m,_ = ep_curves(y,true_mean,true_logZ)
    lap_c,lap_m,_ = laplace_curves(y,true_mean,true_logZ)
    vb_c,vb_m,_ = vb_curves(y,true_mean,true_logZ)
    is_c,is_m,_ = importance_curves(y,true_mean,true_logZ)
    gib_c,gib_m = gibbs_curves(y,true_mean)

    plt.figure(figsize=(6,5))

    
    
    plt.loglog(gib_c,gib_m,label="Gibbs")
    plt.loglog(ep_c,ep_m,label="EP")
    plt.loglog(vb_c,vb_m,label="VB")
    plt.loglog(lap_c,lap_m,label="Laplace")
    plt.loglog(is_c,is_m,label="Importance")
    

    plt.xlabel("FLOPs")
    plt.ylabel("Error")
    plt.title("Posterior mean")

    plt.legend()

    plt.show()


posterior_shape_experiment()
figure2b()


