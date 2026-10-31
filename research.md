---
layout: page
title: Research
permalink: /research/
---

# Research

My research develops **Representative Learning (RepL)**, a data-centric framework for distributed learning in which nodes communicate compact representative objects rather than raw data, parameters, or gradients.

This page presents the technical development of RepL through four questions:

1. What is a representative?
2. What information should it preserve?
3. What mathematical form should it take?
4. How should representatives support communication and collaboration?

---

# 1. Representative Learning Architecture

Consider data distributed across $K$ nodes,

$$
\mathcal D=\bigcup_{k=1}^{K}\mathcal D_k,
\quad
\mathcal D_k=\{(x_i,y_i):i\in I_k\}.
$$

Node $k$ constructs a representative dataset

$$
\tilde{\mathcal D}_k=\mathcal R_k(\mathcal D_k),
$$

where $\mathcal R_k$ is a representative operator designed to preserve selected information from $\mathcal D_k$.

The communicated collection is

$$
\tilde{\mathcal D}=\bigcup_{k=1}^{K}\tilde{\mathcal D}_k.
$$

In a centralized architecture, a coordinating unit learns from $\tilde{\mathcal D}$ in place of the inaccessible distributed data. In a decentralized architecture, node $k$ combines its local data with representatives received from other nodes.

<p align="center">
  <img src="assets/representative_arch.png"
       alt="Representative Learning architecture"
       width="800">
</p>

The central design principle is to separate two decisions:

- **Construction:** what information should the sender preserve?
- **Use:** how should the receiver evaluate and use that information?

---

# 2. Point Representatives

A point representative has the form

$$
\tilde{\mathcal D}_k=(n_k,\tilde{x}_k,\tilde{y}_k),
$$

where $(\tilde{x}_k,\tilde{y}_k)$ is a pseudo-observation and $n_k$ is its weight.

More generally, a node may construct several representatives,

$$
\tilde{\mathcal D}_k = \{(n_{kh},\tilde{x}_{kh},\tilde{y}_{kh})\}_{h=1}^{r_k},
$$

to retain variation across local subpartitions.

## 2.1 Mean Representative

The **Mean Representative (MR)** preserves local feature and response means. It is model-free and yields an unbiased estimator for linear models, but it does not adapt to the current model during learning.

## 2.2 Transformed Mean Representative

The **Transformed Mean Representative (TMR)** modifies mean-based construction for nonlinear generalized linear models:

$$
\tilde{x}_k=\bar{x}_k,\quad\tilde{y}_k=G\{\frac{1}{n_k}\sum_{i\in I_k}g(y_i)\},
$$

where $g$ is the link function and $G=g^{-1}$.

TMR reduces bias induced by nonlinear links while retaining a simple one-shot construction. Like MR, it does not adapt as the global model changes.

## 2.3 Score-Matching Representative

Let

$$
\eta_i=x_i^\top\beta,\quad r_i=y_i-G(\eta_i).
$$

The **Score-Matching Representative (SMR)** preserves the local score contribution:

$$
\sum_{i\in I_k}\nu(\eta_i)r_i x_i=n_k\nu(\tilde{\eta}_k)\tilde r_k \tilde{x}_k.
$$

SMR is model-adaptive and preserves local gradient direction. The original score-matching system, however, can be underdetermined, so its representative may not be unique.

## 2.4 Response-Aided SMR

The **Response-Aided Score-Matching Representative (RASMR)** refines local blocks using response- and model-related information before score matching.

The refined partition improves representative fidelity under heterogeneous local blocks and supports representative-based model and variable selection. The cost is an increase in the number of representatives.

## 2.5 Anchored-SMR

The **Anchored Score-Matching Representative** supplements score matching with an intercept-matching condition. This produces a unique analytic representative.

Its covariate component has the barycentric form

$$
\tilde{x}_k=\frac{\sum_{i\in I_k}h(\eta_i,y_i)x_i}{\sum_{i\in I_k}h(\eta_i,y_i)},
$$

where

$$
h(\eta_i,y_i)=\nu(\eta_i)\{y_i-G(\eta_i)\}.
$$

Anchored-SMR preserves model-adaptive score information while improving identifiability, geometric representativeness, and convergence stability.

Because Anchored-SMR encodes directional information at the current model, it is used with a one-step update rather than full minimization:

$$
\tilde{\beta}^{(t+1)}=\tilde{\beta}^{(t)}-\gamma S(\tilde{\beta}^{(t)};\tilde{\mathcal D}^{(t)}).
$$

<p align="center">
  <img src="assets/flowchart_ASMR.png"
       alt="TMR initialization and Anchored-SMR updating"
       width="320">
</p>

## 2.6 Barycentric Representativeness

Across several constructions, the representative location takes the form

$$
\tilde{x}=\frac{\sum_i w_i x_i}{\sum_i w_i}.
$$

The weights encode the information to be preserved. This motivates the **barycentric representativeness principle**:

> A representative is often a weighted center of local data, or of a representation of local data, where the weights encode the preservation objective.

---

# 3. Clouded Representatives and Lifted ERM

A point representative may not adequately preserve local variability or geometry. We therefore extend the representative to a probability distribution:

$$
\tilde{\mathcal D}_k=(n_k,Q_k).
$$

When $Q_k$ belongs to a Gaussian family, the representative has the form

$$
\tilde{\mathcal D}_k=(n_k,\tilde{x}_k,\tilde{\Sigma}_k,\tilde{y}_k),
$$

which we call a **clouded representative**.

<p align="center">
  <img src="assets/point_clouded_rep.png"
       alt="Point and clouded representatives"
       width="850">
</p>

The location $\tilde{x}_k$ describes where the representative information is centered, while $\tilde{\Sigma}_k$ carries local variation and geometry.

## 3.1 Lifted Empirical Risk

To place ordinary observations and clouded representatives in a common framework, each observation is lifted into a local probability distribution:

$$
(x_i,y_i)\longrightarrow Q_i.
$$

For a loss $L(\theta;x,y)$, the lifted loss is

$$
L(\theta;Q_i)=\mathbb E_{(X,Y)\sim Q_i}L(\theta;X,Y).
$$

The resulting **Lifted Empirical Risk Minimization** framework evaluates the original loss over local probability clouds. Point-valued learning is recovered when the clouds degenerate to point masses.

Gaussian lifting can also smooth a nonsmooth objective without replacing the original loss by an unrelated surrogate.

## 3.2 Clouded Representatives for Linear Quantile Regression

For linear quantile regression, lift each covariate to

$$
X_i\sim\mathcal N(x_i,s^2I).
$$

Let

$$
r_i=y_i-x_i^\top\beta-\beta_0,\quad
s_\beta=s\|\beta\|,\quad
a_i=\frac{r_i}{s_\beta},\quad
w_i=\tau-\Phi(-a_i).
$$

The Gaussian-lifted quantile risk has the closed form

$$
R_{Q_n}(\theta)=\frac{1}{n}\sum_{i=1}^{n}[r_iw_i+s_\beta\phi(a_i)].
$$

For a nondegenerate cloud, the lifted objective is smooth. As $s\rightarrow0$, it recovers classical quantile regression.

### Clouded Score Matching

The **Clouded Score-Matching Representative (Clouded-SMR)** matches the gradient of the representative lifted risk to the average local lifted gradient at an anchor parameter $\theta^\circ$.

Its location is

$$
\tilde{x}_k=\frac{\sum_{i\in I_k}w_i^\circ x_i}{\sum_{i\in I_k}w_i^\circ}.
$$

The location and response are uniquely determined by score matching, while the covariance is selected to preserve local second-order structure subject to the score-matching constraints.

Subpartitioning observations by the sign of $w_i^\circ$ prevents weight cancellation. Each resulting representative remains a convex combination of observations within its subpartition.

## 3.3 Clouded Representatives for Distributed SVM

For a linear SVM, lift each covariate to

$$
X_i\sim\mathcal N(x_i,\Sigma).
$$

The lifted hinge loss is

$$
\tilde{\ell}_i(w,b)=\mathbb E[\max\{0,1-y_i(w^\top X_i+b)\}].
$$

Let

$$
\nu_i=y_i(w^\top x_i+b),\quad s^2=w^\top\Sigma w,\quad a_i=\frac{1-\nu_i}{s}.
$$

Then

$$
\tilde{\ell}_i(w,b)=(1-\nu_i)\Phi(a_i)+s\phi(a_i).
$$

The lifted loss is smooth for nondegenerate covariance, remains convex, and converges to the ordinary hinge loss as $\Sigma\rightarrow0$.

<p align="center">
  <img src="assets/svm_toy2.png"
       alt="Distributed SVM under heterogeneous local data"
       width="680">
</p>

Representatives are constructed separately for each class:

$$
\tilde{\mathcal D}_{k,c}=(n_{k,c},c,\tilde{x}_{k,c},\tilde{\Sigma}_{k,c}).
$$

- **Clouded MR** preserves class-wise location and covariance without depending on the current SVM parameters.
- **Clouded-SMR** additionally matches the class-wise gradient of the lifted SVM objective and is reconstructed as the model changes.

A receiving node augments its local lifted objective with clouded representatives received from other
