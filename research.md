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

# 4. Structured Representatives

Representative Learning is not restricted to finite-dimensional tabular observations. For structured data, the representative itself should retain a data-compatible structure while preserving information required for downstream learning.

## 4.1 Distributed Function-on-Scalar Regression: Rep-FoS

**Distributed Function-on-Scalar Regression via Representative Learning (Rep-FoS)**  

Rep-FoS considers distributed function-on-scalar regression with sparse and irregular functional responses.

For subject $i$,

$$
Y_i(t)=\alpha(t)+X_i^\top\boldsymbol{\beta}(t)+\varepsilon_i(t),
$$

where $Y_i(t)$ may be observed only at subject-specific time points

$$
T_{i,1},\ldots,T_{i,M_i}.
$$

The local data on node $h$ are

$$
\mathcal D_h=\{(X_i,\{(T_{i,m},Y_{i,m})\}_{m=1}^{M_i}):i\in\mathcal I_h\}.
$$

### Kernel-Weighted Local Objective

At each node, estimation is based on the kernel-weighted functional least-squares objective

$$
L_h(\alpha,\beta)=\int_0^1\sum_{i\in\mathcal I_h}\sum_{m=1}^{M_i}[Y_{i,m}-\alpha(t)-x_i^\top\beta(t)]^2 K_b(T_{i,m}-t)dt.
$$

The kernel weights allow sparse and irregular observations to contribute locally around each evaluation point $t$.

### Mean Representative Initialization

Rep-FoS first constructs kernel-weighted mean representatives using

$$
S_{0h}(t)=\sum_{i\in\mathcal I_h}\sum_{m=1}^{M_i}K_b(T_{i,m}-t),
$$

together with the corresponding kernel-weighted covariate and response means.

Each node communicates only

$$
\{S_{0h}(t),\bar X_h(t),\bar Y_h(t)\}.
$$

The central server aggregates these quantities to obtain an MR estimate

$$
\beta_{\mathrm{Rep-FoS}}^{(0)}(t)=\widehat{\beta}_{\mathrm{MR}}(t).
$$

Thus, MR provides a one-round initialization before model-adaptive score matching begins.

### Functional Score Matching

The central idea of Rep-FoS is to construct node-level representatives whose score approximates the score of the original sparse functional observations:

$$
U_h^{\mathrm{rep}}(\theta)\approx U_h(\theta).
$$

Consequently,

$$
\sum_{h=1}^{H}U_h^{\mathrm{rep}}(\theta)\approx\sum_{h=1}^{H}U_h(\theta).
$$

At an evaluation point $t$, the representative covariate again has a weighted-average structure,

$$
\tilde X_h(t)=\frac{\sum_{i\in\mathcal I_h}\sum_{m=1}^{M_i}K_b(T_{i,m}-t)[\eta_i(t)-Y_{i,m}(t)]
X_i(t)}{\sum_{i\in\mathcal I_h}\sum_{m=1}^{M_i}K_b(T_{i,m}-t)[\eta_i(t)-Y_{i,m}(t)]}.
$$

This provides another instance of the barycentric representativeness principle, now with weights determined jointly by **time localization** through the kernel and **model information** through the score residual.

### Distributed Updating

Rep-FoS proceeds iteratively:

1. initialize the coefficient functions using MR;
2. broadcast the current coefficient estimate to the computing nodes;
3. construct local score-matching representatives;
4. transmit only representative quantities to the central server;
5. update the global functional coefficient estimate;
6. repeat.

Raw subject-level functional observations remain local throughout the procedure.

Rep-FoS therefore extends representative learning in two directions simultaneously: from finite-dimensional responses to functional responses, and from static representatives to representatives indexed by the functional domain and updated with the current coefficient estimate.

## 4.2 Projection-Based Representatives for Images

**Projection-Based Representative Learning for Image Classification in Heterogeneous Distributed Systems**  

Image data create a different challenge for Representative Learning.

Direct pixel averaging remains model-independent but may destroy important image structure. Latent-feature communication may retain task-relevant structure, but the communicated object is tied to a particular encoder or model architecture.

PBR takes an intermediate route:

> **construct in a structured representation space, communicate in the original image domain.**

### Local Tucker Representation

Within each response class $c$, node $k$ computes a local Tucker decomposition

$$
x_i\approx\mathcal G_i\times_1 U_{k,c,1}\times_2 U_{k,c,2}.
$$

The Tucker factors and individual cores remain local. They define a data-derived coordinate system for representative construction and are not themselves communicated.

Because substantial heterogeneity may remain within a class, Tucker cores are further divided into fine clusters.

### Tucker-Induced Representative

For cluster $q$, let

$$
\Delta=\{w:w_i\ge 0,\;\sum_{i=1}^{n_q}w_i=1\}.
$$

PBR constructs the representative core

$$
\tilde{\mathcal G}_q(w)=\sum_{i=1}^{n_q}w_i\mathcal G_i
$$

and reconstructs the representative image

$$
\tilde{x}_q(w)=\tilde{\mathcal G}_q(w)\times_1U_{q,1}\times_2U_{q,2}.
$$

The simplex constraint keeps the representative explicitly connected to the local image cluster.

### Predictive Matching

Geometric structure alone is not sufficient. PBR also requires the representative to reproduce the predictive behavior of the observations it represents.

At communication round $r$, define

$$
D_q^{(r)}(w)=\|f_{\theta_q^{(r-1)}}(\tilde{x}_q(w))-\sum_{i=1}^{n_q}w_i f_{\theta_q^{(r-1)}}(x_i)\|_2^2.
$$

The target is therefore not a one-hot class label. Instead, the representative is matched to the actual predictive behavior of the current local model.

### PBR Objective

Starting from equal weights

$$
w_q^{(0)}=(\frac1{n_q},\ldots,\frac1{n_q})^\top,
$$

PBR solves

$$
w_q^{(r)}=\arg\min_{w\in\Delta}\{D_q^{(r)}(w)+\gamma_q(w)(w-w_q^{(0)})^\top A_q (w-w_q^{(0)})\},
$$

where $A_q$ is the Tucker-core Gram matrix.

The two terms have distinct roles:

- **predictive matching** adapts the representative to the current learning task;
- **Tucker-space anchoring** controls departure from the local image geometry.

The final communicated object remains an image,

$$
\tilde{x}_q^{(r)}=\tilde{x}_q(w_q^{(r)}).
$$

Thus, model information affects representative construction without becoming the communication format.

### Representative Weight and Memory

PBR also assigns each communicated pseudo-image an effective statistical weight. Historical representatives are retained with geometric temporal decay,

$$
r(\tau)=(1-e^{-1})e^{-\tau}.
$$

This creates a finite representative memory in which newer representatives have larger individual influence while older representatives decay and are eventually removed.

A receiving node therefore does not need to replace one representative by another at every communication round. It can maintain a weighted history of communicated information.

### Local-Representative Mixture Training

Local data and received representatives form separate training streams,

$$
L_k^{(b)}(\theta_k)=\frac{1-\alpha_k}{B_L}\sum_{i\in\mathcal B_k^L}\ell(\theta_k;x_i,y_i)+\frac{\alpha_k}{B_R}
\sum_{h\in\mathcal H_k^{(r)}}b_h\ell(\theta_k;\tilde{x}_h,\tilde y_h).
$$

Here the representative weights determine how representative information is distributed within the representative stream, while $\alpha_k$ controls the overall strength of cross-node information relative to local data.

PBR therefore connects four levels of representative design:

$$
\text{data structure}\rightarrow
\text{representative geometry}\rightarrow
\text{predictive adaptation}\rightarrow
\text{representative memory}.
$$
