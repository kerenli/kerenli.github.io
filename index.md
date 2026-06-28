---
layout: page
title: Representative Learning Lab
---
  
## Distributed AI via Representative Learning

Modern AI is increasingly **distributed**: data reside across devices, institutions, and regions.  
This creates fundamental challenges in:

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
  <li><a href="assets/RepL_diagram2.png"">Point Representatives (MR, SMR)</a></li>
  <li><a href="assets/RepL_diagram3.png">Stabilized (TMR, Anchored-SMR)</a></li>
  <li><a href="assets/RepL_diagram4.png">Distributional (MMR, Clouded-SMR)</a></li>
</ul>

<p>
  <a href="#" onclick="openModal('assets/RepL_diagram2.png'); return false;">
    Distributional (MMR, Clouded-SMR)
  </a>
</p>

<!-- Modal -->
<div id="imgModal" class="modal" onclick="closeModal()">
  <span class="close">&times;</span>
  <img class="modal-content" id="modalImg" alt="RepL diagram detail">
</div>

<style>
.modal {
  display: none;
  position: fixed;
  z-index: 1000;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0,0,0,0.8);
}

.modal-content {
  display: block;
  margin: 5% auto;
  max-width: 90%;
  max-height: 85%;
  border: 4px solid white;
  box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.close {
  position: absolute;
  top: 20px;
  right: 35px;
  color: white;
  font-size: 40px;
  font-weight: bold;
  cursor: pointer;
}
</style>

<script>
function openModal(src) {
  document.getElementById('imgModal').style.display = 'block';
  document.getElementById('modalImg').src = src;
}

function closeModal() {
  document.getElementById('imgModal').style.display = 'none';
}
</script>

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
