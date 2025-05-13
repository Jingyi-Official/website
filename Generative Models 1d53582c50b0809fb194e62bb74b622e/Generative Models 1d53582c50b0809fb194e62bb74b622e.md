# Generative Models

# Learning Generative Models

- Models Overview
    - Discriminative Model: Learn a probability distribution p(y|x)
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image.png)
        
    - Generative Model: Learn a probability distribution p(x)
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%201.png)
        
    - Conditional Generative Model: Learn p(x|y)
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%202.png)
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%203.png)
        
- We want to learn a probability distribution $p(x)$ over training samples $x$ such that
    - Generation: If we sample $x_{\text{new}} \sim p(x)$, $x_{\text{new}}$ should look like a real image that from training samples
    - Density estimation: $p(x)$ should be high if $x$ looks like a real image that from training samples, and low otherwise (anomaly detection)
    - Unsupervised representation learning: We should be able to learn what these images have in common, e.g., ears, tail, etc. (features). Learn some underlying hidden structure of the data, for downstream tasks.

# Taxonomy of Generative Models

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%204.png)

# **Subjectives**

A good generative model will create a **diverse** set of outputs that resemble the training data without being exact copies. 

---

# Autoregressive Models - Explicit Density

### Subparts

$$

\begin{aligned}
\mathbf{x} &= (x_1, x_2, x_3, \dots, x_T) \\
p(\mathbf{x}) &= p(x_1, x_2, x_3, \dots, x_T) \\
&= p(x_1) \, p(x_2 \mid x_1) \, p(x_3 \mid x_1, x_2) \cdots p(x_T \mid x_1, \dots, x_{T-1}) \\
&= \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1})
\end{aligned}
$$

$$
\mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{x}_{t-2}, \dots, \mathbf{x}_{t-k})
$$

- Probability of the next subpart given all the previous subparts: Language model, Image generation
- A $k^{th}$ order autoregressive model is a feedforward model which predicts the
next variable $\mathbf{x}_t$ in a time series based on the k previous variables $\mathbf{x}_{t-1}, \mathbf{x}_{t-2}, \dots$
- As in RNNs, parameters are shared across time (same function $f(\dots)$ at each $t$ )
- Autoregressive models make a strong **conditional independence assumption**

### **Different inputs and outputs**

$$
\hat{y}_t = f(\mathbf{x}_t, \mathbf{x}_{t-1}, \dots, \mathbf{x}_{t-k})
$$

- $\hat{y}_t$ is dependent on $\{ \mathbf{x}_i \mid t - k \leq i \leq t \}$ and independent of $\{ \mathbf{x}_i \mid i < t - k \}$

### **Compare with RNNs**

- Recurrent models summarize past information through their hidden state h
- In contrast to auto-regressive models, RNNs have infinite memory
- Autoregressive models are **easier to train (no backprop through time)**

### Multi-Layer Autoregressive Models

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%205.png)

- They effectively perform **multiple causal temporal convolutions**
- They can be combined with residual connections and dilated convolutions

### Classic Autoregressive Models

- PixelRNN
- PixelCNN: masked convolutions - due to conditional independence

### Pros and Cons

- Pros
    - Can explicitly compute likelihood p(x)
    - Explicit likelihood of training data gives good evaluation metric
    - Good samples
- Cons
    - Sequential generation => slow

### Practical

- tutorial: [https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/10-autoregressive-image-modeling.html](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/10-autoregressive-image-modeling.html)
- autoregressive models work on images by modeling the likelihood of a pixel given all previous ones.
- we need height-times-width forward passes through the model
- in autoregressive models, we cannot interpolate between two images because of the lack of a latent representation.

---

# Variational Autoencoders

### Latent Variable Models

$$
\left( f_{\mathbf{w}} : \mathbf{x} \mapsto \mathbf{z} \right) \quad\quad g_{\mathbf{w}} : \mathbf{z} \mapsto \hat{\mathbf{x}}
$$

- LVMs map between observation space $\mathbf{x} \in \mathbb{R}^D$ and latent space $\mathbf{z} \in \mathbb{R}^Q$
- One latent variable gets associated with each data point in the training set
- The latent vectors are smaller than the observations $(Q < D)$ → compression
- Models are linear or non-linear, deterministic or stochastic, with/without encoder
- A little taxonomy
    
    
    |  | Deterministic | Probabilistic |
    | --- | --- | --- |
    | Linear | Principle Component Analysis  | Probabilistic PCA |
    | Non-Linear w/ Encoder  | Autoencoder | Variational Autoencoder |
    | Non-Linear w/o Encoder  |  | Gen. Adv. Networks |

### Autoencoders

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%206.png)

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%207.png)

- Models of this type are called autoencoders as they predict their input as output, ie  outputs are its own inputs.
- In contrast, Generative adversarial networks (next lecture) only have a decoder gw
- The goal is to minimize the reconstruction error (as in PCA)
- If interpolate among the latent space, the AE outputs can be not continual. (kind of like hash)
- Not probabilistic: No way to sample new data from learned model
- Denoising Autoencoders
    - Denoising Autoencoders take noisy inputs and predict the original noise-free data
    - Higher level representations are relatively stable and robust to input corruption
    - Encourages the model to generalize better and capture useful structure
    - Similar to data augmentation (except that the “label” is the noise-free input)
    - Compare denoising and generation:
        - denoising: given a “meaningful” input from $p(x)$, can reconstruct
        - generation: generate new samples independently from random noise or latent space

### Generative Models VS Generative Latent Variable Models

- Generative Models
    - Generative modeling is a broad area of machine learning which deals with models
    of probability distributions $p(x)$ over data points $x$ (e.g., images)
    - The generative model’s task is to capture dependencies / structural regularities in the data (e.g., between pixels in images)
    - Some generative models (e.g., normalizing flows) allow for computing $p(x)$
    - Others (e.g., VAEs) only approximate p(x), but allow to draw samples from $p(x)$
- Generative latent variable models
    
    $$
    p(\mathbf{x}) = \int_{\mathbf{z}} \underbrace{p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z})}_{\text{Generative Process}} \, d\mathbf{z} = \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \left[ p(\mathbf{x} \mid \mathbf{z}) \right]
    $$
    
    - capture the structure in latent variables
    - Generative latent variable models often consider a simple Bayesian model
    - $p(\mathbf{z})$ is the prior over the latent variable $\mathbf{z} \in \mathbb{R}^Q$
    - $p(\mathbf{x} \mid \mathbf{z})$ is the likelihood (= decoder that transforms $\mathbf{z}$ into a distribution over $\mathbf{x}$)
    - $p(\mathbf{x})$ is the marginal of the joint distribution $p(\mathbf{x}, \mathbf{z})$
    - The goal is to maximize $p(\mathbf{x})$ for a given dataset X by learning the two models 
    and p(x|z) such that the latent variables z best capture the latent structure of the data

### Variational Autoencoders

So far, we have discussed deterministic latent variables. We will now take a probabilistic perspective on latent variable models with autoencoding properties.

- from Autoencoder to VAE
    - **autoencoder**
        - each position at **$\mathbf{z}$** represents a feature, and the value indicates the strength or intensity of that feature
        - Determine a **specific value** of $\mathbf{z}$
        - The values of $\mathbf{z}$ are **discontinuous**, somewhat similar to Instant-NGP downside
        - Therefore, the autoencoder **cannot produce meaningful results for interpolated $\mathbf{z}$  values**
    - VAE
        - Aim to enhance the expressiveness of $\mathbf{z}$ for generative purposes
        - Model **$\mathbf{z}$** as a **distribution**
- **Probabilistic spin on autoencoders:**
    1. Learn latent features $\mathbf{z}$ from raw data
    2. **Sample** from the model to generate new data

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%208.png)

- **Core idea of VAE**
    - Assumption:
        - We assume the prior model $p({z})$ to be samplable and computable
        - We assume the likelihood model $p(x|z)$ to be computable
        - In other words, we can sample from $p(z)$ and we can compute the probability
        mass/density of $p(z)$ and $p(x|z)$ for any given $x$ and $z$
    - Assume training data $\{ x^{(i)} \}_{i=1}^N$ is generated from an unobserved (latent) representation $\mathbf{z}$.
    - Assume simple prior $p(z)$, e.g. Gaussian
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%209.png)
        
    - Represent $p(x|z)$ with a neural network (Similar to decoder from autencoder)
        - Decoder must be probabilistic: Decoder inputs $z$, outputs mean $\mu_{x \mid z}$ and (diagonal) covariance $\Sigma_{x \mid z}$
        - Sample $x$ from Gaussian with mean $\mu_{x \mid z}$ and (diagonal) covariance  $\Sigma_{x \mid z}$
        - How to train this model? If we could observe the $z$ for each $x$, then could train a conditional generative model  $p(x|z)$
- **Training: maximize likelihood of data**
    - We don’t observe $z$, so need to marginalize. Consider the Bayesian model of the data $\mathbf{x} \in \mathbb{R}^D$
    
    $$
    p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \int p(\mathbf{z}) \, p(\mathbf{x} \mid \mathbf{z}) \, d\mathbf{z} = \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \left[ p(\mathbf{x} \mid \mathbf{z}) \right]
    $$
    
    - **Method 1**
        - Directly computing and maximizing the likelihood $p(\boldsymbol{x})$ is difficult because it involves integrating out all latent variables $\boldsymbol{z}$ in Equation, which is **intractable for complex models**.
            - Directly computing and maximizing the likelihood $p(\boldsymbol{x})$ is difficult because it involves having access to a ground truth latent encoder ${p(\boldsymbol{z} \mid \boldsymbol{x})}$ in Equation.
                
                ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2010.png)
                
                ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2011.png)
                
                ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2012.png)
                
        - Then using other tricks
            
            $$
            
            \begin{align*}\log p(\boldsymbol{x}) &= \log \int p(\boldsymbol{x}, \boldsymbol{z}) \, d\boldsymbol{z} && \text{(Apply Equation 1)} \\&= \log \int \frac{p(\boldsymbol{x}, \boldsymbol{z}) \, q_\phi(\boldsymbol{z}|\boldsymbol{x})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \, d\boldsymbol{z} && \text{(Multiply by 1 = } \frac{q_\phi(\boldsymbol{z}|\boldsymbol{x})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \text{)} \\&= \log \, \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && \text{(Definition of Expectation)} \\&\geq \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && \text{(Apply Jensen's Inequality)}\end{align*}
            $$
            
            - However, this does not supply us much useful information about what is actually going on underneath the hood; crucially,
            - this proof gives **no intuition on exactly why the ELBO is actually a lower bound of the evidence**, as Jensen’s Inequality handwaves it away.
            - simply knowing that the ELBO is truly a lower bound of the data **does not really tell us why we want to maximize it as an objective**. i.e. why optimizing the ELBO is an appropriate objective at all.
    - **Method 2: Try Bayes’ Rule**
        
        $$
        p_\theta(x) = \frac{p_\theta(x \mid z) \, p_\theta(z)}{p_\theta(z \mid x)} \approx \frac{p_\theta(x \mid z) \, p_\theta(z)}{q_\phi(z \mid x)}
        $$
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/79326243-440c-4e0b-a0ea-824a3391386d.png)
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/faa889b6-0a7d-4dc0-a389-2f09bc453c3b.png)
        
        ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2013.png)
        
        - Solution:
            - Train another network (encoder) that learns $q_\phi(z \mid x) \approx p_\theta(z \mid x)$
            - Jointly train encoder $q$ and decoder $p$ to **maximize the variational lower bound** on the data likelihood
                
                ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2014.png)
                
            - Decoder network: inputs latent code $z$, gives distribution over data $x$
            - Encoder network: inputs data $x$, gives distribution over latent codes $z$
        - obtain distribution of observed data $\boldsymbol{x}$  by **chain rule of probability**
        - better understand the relationship between the evidence and the ELBO
            - Bayes’ Rule + Split up using rules for logarithms; Then we can wrap in an expectation since it doesn’t depend on z
                
                $$
                \begin{align*}
                \log p_\theta(\mathbf{x}) 
                &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x}) \right] \\
                &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[
                \log \frac{p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) q_\phi(\mathbf{z} \mid \mathbf{x})}{q_\phi(\mathbf{z} \mid \mathbf{x}) p_\theta(\mathbf{z} \mid \mathbf{x})}
                \right] \\
                &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[
                \log p_\theta(\mathbf{x} \mid \mathbf{z}) - \log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p(\mathbf{z})} + \log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})}
                \right] \\
                &= \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x} \mid \mathbf{z}) \right] 
                - D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})) 
                + D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{x})) \\
                &\geq \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x} \mid \mathbf{z}) \right] 
                - D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}))
                \end{align*}
                $$
                
            - another view
                
                $$
                \begin{align*}\log p(\boldsymbol{x}) &= \log p(\boldsymbol{x}) \int q_\phi(\boldsymbol{z}|\boldsymbol{x}) \, d\boldsymbol{z} && \text{(Multiply by 1 = } \int q_\phi(\boldsymbol{z}|\boldsymbol{x}) \, d\boldsymbol{z} \text{)} \\&= \int q_\phi(\boldsymbol{z}|\boldsymbol{x}) (\log p(\boldsymbol{x})) \, d\boldsymbol{z} && \text{(Bring evidence into integral)} \\&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} [\log p(\boldsymbol{x})] && \text{(Definition of Expectation)} \\&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z}|\boldsymbol{x})} \right] && \text{(Apply Equation 2)} \\&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z}) q_\phi(\boldsymbol{z}|\boldsymbol{x})}{p(\boldsymbol{z}|\boldsymbol{x}) q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && \text{(Multiply by 1 = } \frac{q_\phi(\boldsymbol{z}|\boldsymbol{x})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \text{)} \\&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] + \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{q_\phi(\boldsymbol{z}|\boldsymbol{x})}{p(\boldsymbol{z}|\boldsymbol{x})} \right] && \text{(Split the Expectation)} \\&= \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] + D_{KL}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \parallel p(\boldsymbol{z}|\boldsymbol{x})) && \text{(Definition of KL Divergence)} \\&\geq \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z}|\boldsymbol{x})} \right] && \text{(KL Divergence always } \geq 0)\end{align*}
                $$
                
            - evidence is equal to the ELBO plus the KL Divergence (which is non-negative) between the approximate posterior qφ(z|x) and the true posterior p(z|x). -> the value of the ELBO can never exceed the evidence.
            - why we seek to maximize the ELBO
                - want to optimize the parameters of our variational posterior qφ(z|x) to exactly match the true posterior distribution p(z|x), which is achieved by minimizing their KL Divergence (ideally to zero). Unfortunately, it is intractable to minimize this KL Divergence term directly, as we do not have access to the ground truth p(z|x) distribution.
                - notice that on the left hand side of Equation 15, the likelihood of our data (and therefore our evidence term log p(x)) is always a constant with respect to φ, as it is computed by marginalizing out all latents z from the joint distribution p(x, z) and does not depend on φ whatsoever.
                - Since the ELBO and KL Divergence terms sum up to a constant, any maximization of the ELBO term with respect to φ necessarily invokes an equal minimization of the KL Divergence term. Thus, the ELBO can be maximized as a proxy for learning how to perfectly model the true latent posterior distribution; the more we optimize the ELBO, the closer our approximate posterior gets to the true posterior.
                - Additionally, once trained, the ELBO can be used to estimate the likelihood of observed or generated data as well, since it is learned to approximate the model evidence log p(x).
- Training: process
    
    ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2015.png)
    
    1. Run input data through encoder to get a distribution over latent codes
    2. Encoder output should match the prior $p(\mathbf{z})$：encourange latent distribution close to Normal
        1. Closed form solution when 
        
        $q_\phi$ is diagonal Gaussian and $p$ is unit Gaussian
        (Assume $\mathbf{z}$ has dimension J)
        
        $$
        \begin{align*}- D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})) &= \int_{\mathbf{z}} q_\phi(\mathbf{z} \mid \mathbf{x}) \log \frac{p(\mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \, d\mathbf{z} \\&= \int_{\mathbf{z}} \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_{z|x}, \boldsymbol{\Sigma}_{z|x}) \log \frac{\mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})}{\mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_{z|x}, \boldsymbol{\Sigma}_{z|x})} \, d\mathbf{z} \\&= \sum_{j=1}^{J} \left( 1 + \log \left( (\boldsymbol{\Sigma}_{z|x})_j^2 \right) - (\boldsymbol{\mu}_{z|x})_j^2 - (\boldsymbol{\Sigma}_{z|x})_j^2 \right)\end{align*}
        $$
        
    3. Sample code $\mathbf{z}$ from encoder output
    4. Run sampled code through decoder to get a distribution over data samples
    5. Original input data should be likely under the distribution output from (4)!
    6. Can sample a reconstruction from (4)
- Generation
    
    ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2016.png)
    
    1. Sample z from prior $p(z)$
    2. Run sampled $z$ through decoder to get distribution over data $x$
    3. Sample from distribution in (2) to generate data
- loss
    - **Gaussian likelihood**: Measures the model’s reconstruction confidence using a Gaussian distribution. The generated image is treated as the mean of the distribution, and the likelihood of the original image under this distribution is computed. A higher probability indicates greater similarity.
    - **KL divergence**: Computes the difference in log-probability of the sample z under two distributions; i.e., the divergence between them.
- The diagonal prior on $p(z)$ causes dimensions of $z$ to be independent → “Disentangling factors of variation”
    
    ![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2017.png)
    
- Pros and Cons
    - Pros
        - Principled approach to generative models
        - Allows inference of $q(z|x)$, can be useful feature representation for other tasks
    - Cons
        - Maximizes lower bound of likelihood: okay, but not as good evaluation as
        PixelRNN/PixelCNN
        - Samples blurrier and lower quality compared to state-of-the-art (GANs)
- Challenges
    - More flexible approximations, e.g. richer approximate posterior instead of diagonal Gaussian, e.g., Gaussian Mixture Models (GMMs)
    - Incorporating structure in latent variables, e.g., Categorical Distributions

## VAE+Autoregressive

Considering the pros and cons of both VAE and autoregressive models, we would like to combine them and get the best of both worlds.

---

# Generative Adversarial Networks

### Theory

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2018.png)

- Implicit model: Generative Adversarial Networks give up on modeling p(x), but **allow us to
draw samples from p(x)**
- Let $\mathbf{x} \in \mathbb{R}^D$ denote an observation, and $p(\mathbf{z})$ a prior over latent variables $\mathbf{z} \in \mathbb{R}^Q$.
    - Generator $G_{\mathbf{w}_G} : \mathbb{R}^Q \mapsto \mathbb{R}^D$ that captures the data distribution, denote the generator network with induced distribution $p_{\text{model}}$
    - Discriminator $D_{\mathbf{w}_D} : \mathbb{R}^D \mapsto [0, 1]$  that estimates if a sample cam from the data distribution, denote the discriminator network which outputs a probability
    - General Idea: they use an **adversarial process** in which two models (“players”) are
    **trained simultaneously**, also referred to as two-player minimax game with value function $V(D, G)$
        
        $$
        G^*, D^* = \arg \min_G \, \arg \max_D \, V(D, G) \\
        V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})} \left[ \log D(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \left[ \log \left(1 - D(G(\mathbf{z})) \right) \right]
        $$
        
        - The goal of G is to **maximize the probability of D making a mistake** – to fool it
        - We train $D_{\mathbf{w}_D}$ to assign probability one to samples from $p_{data}$ and zero to samples from $p_{model}$, and $G_{\mathbf{w}_G}$ to fool $D_{\mathbf{w}_D}$ such that it assigns probability one to samples from pmodel.

### Training

![image.png](Generative%20Models%201d53582c50b0809fb194e62bb74b622e/image%2019.png)

- In practice, however, we must use **iterative numerical optimization** and optimizing
$D_{\mathbf{w}_D}$ **in the inner loop** to completion is computationally prohibitive and **would lead to
overfitting on finite datasets**. Therefore, we resort to alternating optimization:
    - $k$ steps of optimizing $D$ (typically k 2 {1, . . . , 5})
    - 1 step of optimizing $G$ (using **a small enough learning rate**)

### Theoretical Analysis

### Empirical Analysis

- **Mode Collapse:** the generator learns to produce **high-quality sample with very low variability**, covering only a fraction of $p_{data}$
    - Example
        - The generator learns to fool the discriminator by producing values close to Antarctic temperatures
        - The discriminator can’t distinguish Antarctic temperatures but learns that all Australian temperatures are real
        - The generator learns that it should produce Australian temperatures and abandons the Antarctic mode
        - The discriminator can’t distinguish Australian temperatures but learns that all Antarctic temperatures are real
        - Repeat
    - **Core reason: The generator produces only a limited variety of samples, lacking diversity. During training, as the generator attempts to fool the discriminator, it tends to focus on the subset of data that is easiest to replicate at the moment. Although the generated samples may appear realistic, they exhibit a narrow and incomplete distribution.**
    - Strategies for avoiding mode collapse
        - Encourage diversity: Minibatch discrimination allows the discriminator to
        compare samples across a batch to determine whether the batch is real or fake.
        - Anticipate counterplay: Look into the future, e.g., via unrolling the discriminator,
        and anticipate counterplay when updating generator parameters
        - Experience replay: Hopping back and forth between modes can be minimised by
        showing old fake samples to the discriminator once in a while
        - Multiple GANs: Train multiple GANs and hope that they cover all modes.
        - Optimization Objective: Wasserstein GANs, Gradient penalties, . . .

### Pros and cons

- Pros
    - A wide variety of functions and distributions can be modeled (flexibility)
    - Only backpropagation required for training the model (no sampling)
    - No approximation to the likelihood required as in VAEs
    - Samples often more realistic than those of VAEs (but VAEs progress as well)
- Cons
    - No explicit representation of  $p_{model}$
    - Sample likelihood cannot be evaluated
    - The discriminator and generator must be balanced well during training
    to ensure convergence to pdata and to avoid mode collapse

### Gradient tricks

- Adding a gradient penalty wrt. the gradients of D stabilizes GAN training
    
    $$
    
    V(D, G) = 
    \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})} 
    \left[ \log D(\mathbf{x}) 
    - \lambda \left\| \nabla_{\mathbf{x}} D(\mathbf{x}) \right\|^2 
    \right]
    + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} 
    \left[ \log \left(1 - D(G(\mathbf{z})) \right) \right]
    $$
    

### Classic Models

- DCGAN
    - Replace any pooling layers with strided convolutions (discriminator) and
    fractional strided convolutions for upsampling (generator)
    - Use batch normalization in both the generator and the discriminator
    - Remove fully connected hidden layers for deeper architectures
    - Use ReLU activations in the generator except for the output which uses tanh
    - Use Leaky ReLU activations in the discriminator for all layers
- Wasserstein GAN
    - Low dimensional manifolds in high dimension space often have little overlap
    - The discriminator of a vanilla GAN saturates if there is no overlapping support
    - WGANs uses the **Earth Mover’s distance** which can handle such scenarios
- CycleGAN
    - Learn forward and backward mapping btw. two domains (domains = latents)
    - Use cycle-consistency and adversarial losses to constrain this mapping
- BigGAN
    - Scale class-conditional GANs to ImageNet (5122) without progressive growing
    - Key: more parameters, larger minibatches, orthogonal regularization of G
    - Explore variants of spectral normalization and gradient penalties for D
    - Analyze trade-off between stability (regularization) and performance (FID)
    - Monitor singular values of weight matrices of generator and discriminator
    - Found early stopping leads to better FID scores compared to regularizing D
- StyleGAN / StyleGAN2
    - Complex stochastic variation with different realizations of input noise

---

# Evaluation

### With exact likelihood

### Without exact likelihood

- Frechet inception distance (FID)
    
    $$
    \text{FID} = \| \boldsymbol{\mu}_m - \boldsymbol{\mu}_d \|_2^2 + \mathrm{Tr} \left( \boldsymbol{\Sigma}_m + \boldsymbol{\Sigma}_d - 2 \left( \boldsymbol{\Sigma}_m \boldsymbol{\Sigma}_d \right)^{1/2} \right)
    $$
    
    - compares the distribution of generated images with the distribution of real images based on deeper features of a pre-trained Inception v3 network
    - the Frechet distance between **two multidimensional Gaussian distributions**
    - measures image fidelity but cannot measure or prevent mode collapse

---

# Normalizing Flows

- normalizing flows are commonly **parameter heavy** and therefore **computationally expensive.**
- 
    
    In the previous lectures, we have seen Energy-based models, Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) as example of generative models. However, none of them explicitly learn the probability density function
    
    p(x) of the real input data. While VAEs model a lower bound, energy-based models only implicitly learn the probability density. GANs on the other hand provide us a sampling mechanism for generating new data, without offering a likelihood estimate. The generative model we will look at here, called Normalizing Flows, actually models the true data distribution
    
    p(x)
    
    and provides us with an exact likelihood estimate.
    
- 

---

# Energy-based models

---

# Score-based models

---

# Distances of probability distributions

- KL divergence (maximum likelihood)
    - Autoregressive models.
    - Normalizing flow models.
    - ELBO in Variational autoencoders.
    - Contrastive divergence in energy-based models.
- f -divergences, Wasserstein distances
    - Generative adversarial networks (f-GANs, WGANs)
- Fisher divergence (score matching): denoising score matching, sliced
score matching
    - Energy-based models.
    - Score-based generative models
- Noise-contrastive estimation
    - Energy-based models.

# Evaluation of Generative Models

- Density Estimation or Compression
    - Likelihood：$\mathbb{E}_{p_{\text{data}}} \left[ \log p_\theta(\mathbf{x}) \right]$ Evaluate generalization with likelihoods on test set
- Measures how well the model compresses the data
    - Shannon coding: assign codeword of length
    - Intuition
    - Average code length
- Sample quality
    - Human evaluations
    - Inception Scores
        - Sharpness (S)
        - Diversity (D)
    - Frechet Inception Distance
        - similarities in the feature representations
    - Kernel Inception Distance
- Lossy Compression or Reconstruction
    - Mean Squared Error (MSE)
    - Peak Signal to Noise Ratio (PSNR)
    - Structure Similarity Index (SSIM)
- Disentanglement
    - Beta-VAE metric
    - Factor-VAE metric
    - Mutual Information Gap