

---
layout: page title: Representative Learning
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
\widetilde{\mathcal{D}}_k=\mathcal{R}_k(\mathcal{D}_k),
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
\widetilde{\mathcal{D}}_k = \mathcal{R}_k(\mathcal{D}_k)
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


RepL is developed along three main directions:

### 1. Point Representatives (GLM & Smooth ERM)
- Mean Representative (MR)  
- Score-Matching Representative (SMR)  
- Response-Aided SMR (RASMR)  
- Transformed Mean Representative (TMR)  
- Anchored-SMR  

### 2. Non-Smooth Learning (Beyond GLMs)
- SVM and quantile regression  
- Clouded representatives (distribution-based learning)  

### 3. Extensions
- Functional data analysis  
- Asynchronous distributed learning  
- Collaborative learning frameworks  

---
<h2>RepL Trajectory</h2>
<p align="center">
  <img src="assets/RepL_diagram0.png" width="1000">
</p>

<h3>Explore:</h3>
<ul>
  <li><a href="assets/RepL_diagram2.png"  target="_blank">Point Representatives (MR, SMR)</a></li>
  <li><a href="assets/RepL_diagram3.png"  target="_blank">Stabilized (TMR, Anchored-SMR)</a></li>
  <li><a href="assets/RepL_diagram4.png"  target="_blank">Distributional (MMR, Clouded-SMR)</a></li>
</ul>


---

## Visual Overview

<p align="center">
  <img src="assets/representative_arch.png" width="600">
</p>

<p align="center"><em>Representative Learning (RepL) framework for centralized build </em></p>

---

## Publications

See [Publications](publications.md) for a complete list. Recent work includes:

- *Representative Learning for Distributed Learning with Heterogeneity and Asynchrony* (JCGS)

## Professional Experience
See [Experience & Education](experience.md) for details.

## Talks
See [Talks](talks.md) for selected invited and conference talks.

---
