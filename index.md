---
layout: page 
title: Representative Learning
---
  
# Keren Li

## Representative Learning

I develop **Representative Learning (RepL)**, a data-centric framework for distributed learning.

The central question is simple:

> **If data cannot be shared, what information should we communicate?**

Given a local dataset

$$
\mathcal{D}_k=\{(x_i,y_i):i\in I_k\},
$$

we construct a compact pseudo-dataset

$$
\tilde{\mathcal{D}}_k=\mathcal{R}_k(\mathcal{D}_k),
$$

called a **representative**.

The representative operator $\mathcal{R}_k$ is designed to preserve selected information from the local data. The resulting representative can then be communicated and used for downstream learning.

In short,

$$
\boxed{\text{local data}} \quad\longrightarrow\quad \boxed{\text{representative}} \quad\longrightarrow\quad \boxed{\text{communication and learning}}.
$$

---

- Data heterogeneity (non-IID distributions)
- Communication constraints
- Privacy and data localization
- Asynchronous systems

---

## Birth of Representative Idea

The idea of a representative did not originally come from distributed learning.

During my PhD study, my advisor **Jie Yang** and I were working on D-optimal sampling for large datasets with multinomial logistic models. We discretized continuous variables to construct an initial design before applying the subsampling procedure.

An unexpected phenomenon caught my attention: the simple initial design could perform better than the samples obtained after subsequent updating.

This made me wonder whether discretization was doing something more interesting than merely providing an initial value.

Instead of asking

> **Which observations should we sample?**

I started to ask

> **Given a block of data, can we construct one pseudo-data point that better represents the information contained in the whole block?**

That question became the prototype of the **representative**.

A natural first choice is a center of the local observations, such as their mean. But for statistical learning, geometric closeness alone may not be sufficient. A better representative may need to preserve how the entire data block contributes to model estimation.

This consideration led from **Mean Representatives (MR)** to **Score-Matching Representatives (SMR)**: rather than only matching the center of a data block, construct a pseudo-data point that matches its contribution to the estimating equation.

The representative therefore evolved from a simple data summary to a statistical object designed to preserve selected information from local data.

---

## From Representative to Representative Learning

Now consider a distributed dataset

$$
\mathcal{D}=\bigcup_{k=1}^{K}\mathcal{D}_k,
$$

where node $k$ holds a local dataset $\mathcal{D}_k$ that cannot be directly shared.

Traditional distributed learning often communicates model-derived quantities, such as parameters or gradients.

Representative Learning takes a different route.

Each node constructs

$$
\tilde{\mathcal{D}}_k = \mathcal{R}_k(\mathcal{D}_k)
$$

and communicates the **representative** instead.

<p align="center">
  <img src="assets/representative_arch.png"
       alt="Representative Learning architecture"
       width="800">
</p>

The key distinction is:
> **The sender determines what information should be preserved.  
> The receiver determines how that information should be used.**

This separates the **construction of information** from its **downstream use**.

---

# Why Representative?

### Data-format communication

Model parameters and gradients describe information through a particular model and objective.

Representatives instead preserve selected information in the data space. The receiving node can evaluate the communicated information using its own learning procedure seamlessly.

### Model heterogeneity

Different nodes may use different models, objectives, or internal representations.

Their parameters or gradients may therefore have different meanings. Representatives provide a communication object that does not require the receiving node to maintain the same model state as the sender.

### Persistent information

A gradient primarily describes an update at the current optimization state.

A representative remains an information object after communication. It can be stored, inspected, combined with local data or other representatives, and reused for later learning.

### Local evaluation

Communication does not imply acceptance.

A receiving node can evaluate whether a representative is relevant to its own data, model, and task before determining how the communicated information should influence local learning.

---

# Point Representatives

The first generation of representatives takes the form

$$
\tilde{\mathcal{D}}_k=(n_k,\tilde{x}_k,\tilde{y}_k),
$$

where $(\tilde{x}_k,\tilde{y}_k)$ is a pseudo-data point and $n_k$ records the amount of local data represented by this point.

More generally, a local dataset may be divided into $r_k$ subsets,

$$
\tilde{\mathcal{D}}_k=\{(n_{kh},\tilde{x}_{kh},\tilde{y}_{kh})\}_{h=1}^{r_k}.
$$

This gives a collection of weighted pseudo-data points that can retain more local variation.

Our development of point representatives includes:

**Mean Representative (MR)** 
The simplest construction preserves local feature and response means. It is model-free and yields an unbiased estimator for linear models, but does not account for the nonlinear relationship. It is robust but does not adapt to the current model during learning. 

**Transformed Mean Representative (TMR)** 
TMR extends mean-based construction to nonlinear generalized linear models by averaging the response in the transformed space,

$$
\tilde{y}_k=G\{\frac{1}{n_k}\sum_{i\in I_k} g(y_i)\},
$$

where $g$ is the link function and $G=g^{-1}$. TMR reduces the bias induced by nonlinear links while retaining a simple one-shot construction, but it does not adapt as the global model evolves.

**Score-Matching Representative (SMR)** 
SMR makes representative construction model-adaptive by matching the local score contribution,

$$
\sum_{i\in I_k}\nu(\eta_i)r_i x_i=n_k\nu(\tilde{\eta}_k)\tilde{r}_k\tilde{x}_k.
$$

**Response-Aided Score-Matching Representative (RASMR)** 
RASMR refines SMR by further partitioning heterogeneous local data using response- and model-related information. The refined blocks improve identifiability and stability of score-matching representatives, at the cost of constructing more representatives.

**Anchored Score-Matching Representative (Anchored-SMR)** 
Anchored-SMR resolves the underdetermined score-matching system by adding an intercept-matching condition. This yields a unique closed-form representative whose covariate component has the barycentric form

$$
\tilde{x}_k=\frac{\sum_{i\in I_k}h(\eta_i,y_i)x_i}{\sum_{i\in I_k}h(\eta_i,y_i)},\quad
h(\eta_i,y_i)=\nu(\eta_i)\{y_i-G(\eta_i)\}.
$$

It retains the model-adaptive property of score matching while improving identifiability, geometric representativeness, and stability.

**From means to model-adaptive representatives:**  
MR → TMR introduces nonlinear structure → SMR introduces model/gradient information → RASMR refines local heterogeneity → Anchored-SMR stabilizes and identifies score matching.

<p align="center">
  <img src="assets/flowchart_ASMR.png"
       alt="Representative Learning architecture"
       width="250">
</p>

---

## A Recurring Structure

Across several representative constructions, the representative location has a barycentric form

$$
\tilde{x}=\frac{\sum_i w_i x_i}{\sum_i w_i}.
$$

The weights $w_i$ change according to the information that the representative is designed to preserve.

This motivates a principle that appears repeatedly in our work:

> **Barycentric representativeness principle:** 
> A representative is often a weighted center of local data, or of a representation of the local data, where the weights encode the information to be preserved.

---

# From Points to Clouds

A point representative is compact, but a single pseudo-observation may not adequately retain local variability or geometric structure. We therefore extend the representative from a point to a probability distribution,

$$
\tilde{\mathcal{D}}_k=(n_k,Q_k),
$$

When $Q_k$ belongs to a Gaussian family, the representative takes the form

$$
\tilde{\mathcal{D}}_k=(n_k,\tilde{x}_k,\tilde{\Sigma}_k,\tilde{y}_k),
$$

which we call a **clouded representative**.

<p align="center">
  <img src="assets/point_clouded_rep.png"
       alt="Point and clouded representatives"
       width="450">
</p>

---

## Lifted Empirical Risk

Distributional representatives create a natural question: how should an ordinary observation and a clouded representative enter the same learning problem?

We address this by lifting an observation from a point to a probability distribution,

$$
(x_i,y_i)\quad\rightarrow\quad Q_i.
$$

For a loss $L(\theta;x,y)$, define the lifted loss

$$
L(\theta;Q_i)=\mathbb{E}_{(X,Y)\sim Q_i}L(\theta;X,Y).
$$

This leads to **Lifted Empirical Risk Minimization (Lifted ERM)**. A point observation is recovered when $Q_i$ degenerates to a point mass, while a nondegenerate $Q_i$ retains local distributional information.

### Linear Quantile Regression

For quantile regression, we lift each covariate to

$$
X_i\sim\mathcal N(x_i,s^2I).
$$

Let

$$
r_i=y_i-x_i^\top\beta-\beta_0,\quad s_\beta=s\|\beta\|,\quad a_i=\frac{r_i}{s_\beta}, \quad w_i=\tau-\Phi(-a_i).
$$

The Gaussian-lifted quantile risk has the closed form

$$
R_{Q_n}(\theta)=\frac{1}{n}\sum_{i=1}^n[r_iw_i+s_\beta\phi(a_i)].
$$

Gaussian lifting smooths the check-loss objective for a nondegenerate cloud while recovering classical quantile regression as the cloud scale tends to zero.

We then construct a **Clouded Score-Matching Representative (Clouded-SMR)** by matching the gradient of the representative lifted risk to the average local lifted gradient.

Its representative location again takes a barycentric form,

$$
\tilde{x}_k=\frac{\sum_{i\in I_k}w_i^\circ x_i}{\sum_{i\in I_k}w_i^\circ}.
$$

Thus, the barycentric representativeness principle persists beyond point representatives: the representative remains a weighted local center, while the cloud additionally carries scale and covariance information.

### Support Vector Machines

Distributed SVM becomes difficult when individual nodes observe substantially different local class configurations.

<p align="center">
  <img src="assets/svm_toy2.png"
       alt="Distributed SVM under heterogeneous local data"
       width="450">
</p>

The same lifted framework applies to the nonsmooth hinge loss. We lift

$$
X_i\sim\mathcal N(x_i,\Sigma)
$$

and define the lifted hinge loss

$$
\tilde{\ell}_i(w,b)=\mathbb E[\max\{0,1-y_i(w^\top X_i+b)\}].
$$

Let

$$
\nu_i=y_i(w^\top x_i+b),\quad s^2=w^\top\Sigma w,\quad a_i=\frac{1-\nu_i}{s}.
$$

Then the expectation has the exact closed form

$$
\tilde{\ell}_i(w,b)=(1-\nu_i)\Phi(a_i)+s\phi(a_i).
$$

The Gaussian lifting produces a smooth convex objective for nondegenerate clouds and recovers the ordinary hinge loss as the covariance degenerates to zero.

---

# Beyond Vector-Valued Data

The idea of a representative is not restricted to a vector-valued pseudo-data point.

The same question can be asked for more complex data:

> **What compact object preserves the information needed from a local dataset while remaining useful to another learner?**

This leads to extensions of Representative Learning to structured objects.

### Functional Representatives

For functional data, the representative itself can retain functional structure rather than reducing the local information to a finite-dimensional model update.

### Image Representatives

For image classification, an image-format representative can preserve local image structure while remaining in the original image domain.

### Other Structured Representatives

More generally, representatives may take point, distributional, functional, image, or other structured forms depending on the information that needs to be preserved.

This changes the question from

> *What is the best pseudo-point?*

to the broader question

> **What should a representative preserve, and what form should it take?**

---

# Current Research

Our current work develops Representative Learning along several connected directions.

### Foundations of Representative Learning

Characterize what information a representative can preserve, what information should be preserved for a particular learning task, and how representative operators can be constructed systematically.

### Functional Representative Learning

Develop representative constructions for distributed functional data analysis.

### Representative Learning for Images

Construct representatives that remain in the image domain while preserving information useful for learning under data and model heterogeneity.

### Heterogeneity and Collaboration

Use representatives not only for distributed optimization, but also for measuring differences between local information and determining when collaboration across nodes is useful.

---

# From Distributed Learning to Distributed AI

Representative Learning began with a statistical question:

> **How can one pseudo-data point represent a block of observations?**

In distributed learning, this became a communication question:

> **What information should one node communicate to another?**

For distributed AI, the question becomes broader:

> **How should autonomous learning agents communicate, evaluate, and selectively use knowledge from other agents?**

Our current vision organizes this process around four operations:

$$
\boxed{\text{Represent}}\;\longrightarrow\;\boxed{\text{Communicate}}\;\longrightarrow\;\boxed{\text{Validate}}\;\longrightarrow\;\boxed{\text{Regulate}}.
$$

**Represent.** Construct a compact object that preserves selected information from local data or knowledge.

**Communicate.** Exchange the representative rather than requiring access to raw observations or a common model state.

**Validate.** Evaluate received information relative to the receiver's own data, objective, model, and available evidence.

**Regulate.** Determine how strongly the received information should influence subsequent learning.

Under this view, distributed intelligence does not require every node to converge to a common internal state.

Instead, each learning agent can maintain its own data, model, task, and accumulated knowledge while communicating through representatives.

A representative therefore becomes more than a data-reduction device.

> **It becomes a communication carrier for distributed intelligence.**

---

## Research Trajectory

<p align="center">
  <img src="assets/RepL_diagram0.png"
       alt="Representative Learning research trajectory"
       width="1000">
</p>

The trajectory of this research can be summarized as

$$
\text{data block}\longrightarrow\text{point representative}\longrightarrow\text{clouded and structured representatives}\longrightarrow\text{Representative Learning}\longrightarrow\text{distributed AI}.
$$

---

## Selected Publications

- K. Li and J. Yang. **Score-Matching Representative Approach for Big Data Analysis with Generalized Linear Models.** *Electronic Journal of Statistics*, 2022.

- D. Zheng, K. Li, and J. Yang. **Response-Aided Score-Matching Representative Approaches for Big Data Analysis and Model Selection under Generalized Linear Models.** *Algorithms*, 2024.

- K. Li. **Representative Learning for Distributed Learning with Heterogeneity and Asynchrony.** *Journal of Computational and Graphical Statistics*, 2026.

- J. Casey, Q. Chen, M. Fan, B. Geng, R. Shterenberg, Z. Chen, and K. Li. **Distributed Collaborative Learning with Representative Knowledge Sharing.** *Mathematics*, 2025.

[See all publications →](publications.md)

---

## About

I am **Keren Li**, an Assistant Professor at the University of Alabama at Birmingham. My research interests include statistical machine learning, distributed learning, data heterogeneity, and Representative Learning.

[Research →](research.md) &nbsp;&nbsp;
[Publications →](publications.md) &nbsp;&nbsp;
[Talks →](talks.md) &nbsp;&nbsp;
[Experience & Education →](experience.md)
