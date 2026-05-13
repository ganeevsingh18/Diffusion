# Diffusion

A small experimental repo for learning and understanding diffusion models from

## First Principles

DiffusionFirstPrinciples is a small exploratory notebook where I try to understand diffusion models from first principles.
The notebook starts with samples from a Poisson distribution (2-D for better visualisation and building intuition):

$$
x_0 \sim \text{Poisson}(\lambda)
$$

and gradually transforms them into a unit Normal distribution through a forward diffusion process:

$$
q(x_t \mid x_{t-1}) =\mathcal{N}\left(\sqrt{1-\beta_t}x_{t-1}, \beta_t I\right)
$$

where $\beta_t$ controls the amount of noise added at each timestep.
The reverse process learns to denoise step by step:

$$
p_\theta(x_{t-1} \mid x_t)
$$

The notebook trains a model to approximate this reverse transition using the ELBO objective. In simplified form, the training loss becomes a noise-prediction objective:

$$
\mathcal{L}=\mathbb{E}_{x_0, \epsilon, t} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]
$$

