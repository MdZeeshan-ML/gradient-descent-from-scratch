# Gradient Descent Linear Regression - Mathematical Theory

**Date:** January 28, 2026  
**Student:** Mohammad Zeeshan Hussain

---

## Section 1: Formal Problem Setup (S6E1 Exam Score Prediction)

### Problem Statement

We want to predict a student's exam score $y$ from their characteristics and study behavior $x$.

### Components

**Feature vector** $x \in \mathbb{R}^d$: For each student $i$,

$$x_i = (\text{age}, \text{study\_hours}, \text{class\_attendance}, \text{sleep\_hours}, ...)$$

**Note:** In practice we drop `id` (it doesn't carry signal) and treat the others as features after encoding.

**Target** $y \in \mathbb{R}$:

$$y_i = \text{exam\_score}_i$$

A real-valued exam score for student $i$.

**Dataset:**

$$D = \{(x_i, y_i)\}_{i=1}^{N}$$

where each pair is one row of `train.csv`, with $N \approx 630000$.

We assume the pairs $(x_i, y_i)$ are **i.i.d.** (independent and identically distributed) from some unknown distribution $P(X, Y)$.

**Model** $f_\theta(x)$:

A function that maps features $x$ to a predicted score $\hat{y} = f_\theta(x)$.

**Loss per example:**

$$\ell(f_\theta(x_i), y_i) = (y_i - f_\theta(x_i))^2$$

**Empirical risk (training objective):**

$$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - f_\theta(x_i))^2$$

**Goal:**

$$\theta^\star = \arg\min_{\theta} J(\theta)$$

---

## Section 2: Linear Regression Model

### Model Definition

We choose a linear model:

**Parameters:** $w \in \mathbb{R}^d$, $b \in \mathbb{R}$, $\theta = (w, b)$

**For each sample:**

$$f_\theta(x_i) = w^\top x_i + b$$

**Objective:**

$$J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - (w^\top x_i + b))^2$$

### Prediction and Error

**Prediction:**

$$\hat{y}_i = f_\theta(x_i) = w^\top x_i + b$$

**Error (residual):**

$$e_i = y_i - \hat{y}_i = y_i - (w^\top x_i + b)$$

**Interpretation:**
- If $e_i > 0$: underpredicted (model's prediction too low)
- If $e_i < 0$: overpredicted (model's prediction too high)
- If $e_i = 0$: perfect prediction for that student

### Loss Function

**Per-example loss:**

$$\ell_i = e_i^2$$

**Why square?**
1. Makes all penalties positive (no cancellation)
2. Penalizes large errors more than small ones
3. Gives smooth math for differentiation

**Objective (Mean Squared Error):**

$$J(w, b) = \frac{1}{N} \sum_{i=1}^{N} e_i^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - (w^\top x_i + b))^2$$

---

## Section 3: Gradient Derivation

### Goal

Find:

 $\frac{\partial J}{\partial w}$ (gradient w.r.t. weights)

 $\frac{\partial J}{\partial b}$ (gradient w.r.t. bias)

These tell us how to update parameters to reduce $J$.

---

### Part A: Gradient w.r.t. Bias $b$

**Step 1:** Write $J$ in terms of errors

$$J(w,b) = \frac{1}{N} \sum_{i=1}^{N} e_i^2$$

where $e_i = y_i - (w^\top x_i + b)$.

**Step 2:** Apply chain rule

$$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial (e_i^2)}{\partial b}$$

**Step 3:** Derivative of $e_i^2$ w.r.t. $b$

Use chain rule:

$$\frac{\partial (e_i^2)}{\partial b} = 2e_i \cdot \frac{\partial e_i}{\partial b}$$

Since $e_i = y_i - (w^\top x_i + b)$:

$$\frac{\partial e_i}{\partial b} = -1$$

Therefore:

$$\frac{\partial (e_i^2)}{\partial b} = 2e_i \cdot (-1) = -2e_i$$

**Step 4:** Final gradient

$$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (-2e_i) = -\frac{2}{N} \sum_{i=1}^{N} e_i$$

Dropping constant 2:

$$\frac{\partial J}{\partial b} = -\frac{1}{N} \sum_{i=1}^{N} e_i$$

---

### Part B: Gradient w.r.t. Weights $w$

**Step 1:** Start the same way

$$\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial (e_i^2)}{\partial w}$$

**Step 2:** Chain rule on $e_i^2$

$$\frac{\partial (e_i^2)}{\partial w} = 2e_i \cdot \frac{\partial e_i}{\partial w}$$

**Step 3:** Derivative of $e_i$ w.r.t. $w$

Since $e_i = y_i - (w^\top x_i + b)$:

$$\frac{\partial e_i}{\partial w} = -x_i$$

(Standard vector calculus: derivative of $w^\top x$ w.r.t. $w$ is $x$)

**Step 4:** Combine

$$\frac{\partial (e_i^2)}{\partial w} = 2e_i \cdot (-x_i) = -2e_i x_i$$

**Step 5:** Final gradient

$$\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (-2e_i x_i) = -\frac{2}{N} \sum_{i=1}^{N} e_i x_i$$

Dropping constant 2:

$$\frac{\partial J}{\partial w} = -\frac{1}{N} \sum_{i=1}^{N} e_i x_i$$

---

### Summary: Gradient Formulas

$$\frac{\partial J}{\partial b} = -\frac{1}{N} \sum_{i=1}^{N} e_i$$

$$\frac{\partial J}{\partial w} = -\frac{1}{N} \sum_{i=1}^{N} e_i x_i$$

where $e_i = y_i - (w^\top x_i + b)$.

---

## Section 4: Gradient Descent Algorithm

### Update Rules

Once per iteration:

1. Compute predictions: $\hat{y}_i = w^\top x_i + b$ for all $i$

2. Compute errors: $e_i = y_i - \hat{y}_i$ for all $i$

3. Compute gradients using formulas above

4. Update parameters:

$$b \leftarrow b - \eta \frac{\partial J}{\partial b} = b + \eta \cdot \frac{1}{N} \sum_{i=1}^{N} e_i$$

$$w \leftarrow w - \eta \frac{\partial J}{\partial w} = w + \eta \cdot \frac{1}{N} \sum_{i=1}^{N} e_i x_i$$

where $\eta$ = learning rate (step size)

5. Repeat until $J(w,b)$ stops decreasing (convergence)

---

## Section 5: Feature Normalization (Standardization)

### Why Normalize?

Different features have different scales. Without normalization, gradient descent becomes unstable.

### Formula

For each feature column:

$$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

where:
- $\mu$ = mean of that feature
- $\sigma$ = standard deviation of that feature

After normalization, each feature has mean ≈ 0 and std ≈ 1.

### Steps

1. Compute mean: $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$
2. Compute std: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$
3. Normalize: $x_{\text{norm}} = \frac{x - \mu}{\sigma}$

---

## Key Concepts Explained

### What is i.i.d.?

**Independent and Identically Distributed:**
- **Identically distributed:** Each sample comes from the same distribution
- **Independent:** Knowing one sample gives no info about another

### What does a weight mean?

Weight $w_j$ for feature $j$:

"If feature $j$ increases by 1 unit (holding others constant), predicted score changes by $w_j$ points."

### What does the gradient tell us?

The gradient $\nabla J$ points in the direction of **steepest increase** in loss.

We move **opposite** to the gradient (hence the minus sign in update rule) to go **downhill** toward minimum loss.

### Why the minus sign in $w = w - \eta \nabla_w J$?

| Situation | Gradient Sign | Update | Result |
|-----------|---------------|--------|--------|
| Underpredicting | Negative | $w - (\text{negative}) = w + \text{positive}$ | Increase $w$ ✓ |
| Overpredicting | Positive | $w - (\text{positive})$ | Decrease $w$ ✓ |

The minus ensures we always move toward lower loss.

---

## Final Notes

This derivation is the foundation of all modern deep learning. Neural networks use the exact same principles, just with more layers and non-linear activation functions.

The chain rule we used here extends to backpropagation in deep networks.
