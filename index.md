

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

- Data heterogeneity (non-IID distributions)
- Communication constraints
- Privacy and data localization
- Asynchronous systems

---

## A Data-Centric Paradigm

We develop **Representative Learning (RepL)**, a new framework for distributed machine learning.

Instead of sharing:
- model parameters  
- gradients  

RepL enables nodes to communicate **structured pseudo-data (“representatives”)**, preserving statistical structure while enabling efficient global learning.

---

## Why RepL?

- ✅ Communication-efficient (compact representations)  
- ✅ Model-agnostic (works across methods)  
- ✅ Privacy-aware (no raw data sharing)  
- ✅ Robust to heterogeneity  
- ✅ Naturally supports asynchronous systems  

---

## Research Overview

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
