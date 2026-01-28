# Gradient Descent Linear Regression from Scratch

**Author:** Mohammad Zeeshan Hussain  
**Date:** January 28, 2026  
**Project:** Kaggle Playground Series S6E1 - Student Exam Score Prediction

---

## Project Overview

### Objective

We want to predict a student's exam score $y$ from their characteristics and study behavior $x$.

### Dataset

- **Source:** Kaggle Playground Series S6E1
- **Task:** Predict student exam scores from behavioral and demographic features
- **Training samples:** ~$630,000$ students
- **Features used:** `age`, `study_hours`, `class_attendance`, `sleep_hours` ($4$ numeric features)
- **Target:** `exam_score` (continuous, range $19.6 - 100.0$)

---

## Implementation

### Algorithm

Implemented **batch gradient descent** for linear regression from scratch using NumPy.

**Model:**
```
score = w₁·age + w₂·study_hours + w₃·class_attendance + w₄·sleep_hours + b
```

**Loss function:** Mean Squared Error (MSE)

**Optimization:** Gradient descent with learning rate $η = 0.01$

**Update rules:**

$$w \leftarrow w - \eta \cdot \nabla_w J$$

$$b \leftarrow b - \eta \cdot \nabla_b J$$



### Key Functions

1. `load_data()` - Load and extract features/target from CSV

2. `normalize_features()` - Standardize features (mean=$0$, std=$1$)

3. `predict()` - Compute predictions: 

    $$ŷ = Xw + b$$

4. `compute_loss()` - Calculate MSE

5. `compute_gradients()` - Compute $\nabla J$ using analytical gradient formulas


6. `train_gradient_descent()` - Main training loop

---

## Results

### Training Performance

- **Training samples:** $10,000$ (subset for faster iteration)
- **Epochs:** $300$
- **Initial loss:** $4264.6415$
- **Final loss:** $122.9191$
- **Final RMSE:** $11.09$

### Learned Parameters

```python
Weights:  [ 0.07069978 13.20355199  5.44468805  2.40144017] 
Bias: 59.4422
```

### **Interpretation:**
---
### What each weight means:

1. **Bias** = $59.62$ : Baseline score when all normalized features = $0$ (i.e., when student is exactly average in all dimensions)

2. **study_hours** = $13.07$ *(LARGEST)*:

- Most important factor

- Each extra hour of study (above average) adds ~$13$ points to predicted score

- This matches intuition: studying is the strongest predictor of exam performance 

3. **class_attendance** = $5.25$ *(second largest)*:

- Important but less than study hours

- Higher attendance → higher scores

- Makes sense: attending class helps 

4. **sleep_hours** = $2.17$ *(positive)*:

- More sleep → slightly better scores

- Smaller effect, but still helps

- Aligns with "well-rested students perform better" 

5. age = $-0.23$ (slightly negative):

- Older students score very slightly lower

- Effect is tiny (almost negligible)

- Could be noise or subtle correlation in this dataset


### Model Insights

**Convergence pattern:**
- Epochs$ 0-100$: Loss drops from $4280 → 665$ (rapid learning)
- Epochs $100-200$: Loss drops from $665 → 185$ (slower)
- Epochs $200-300$: Loss drops from $185 → 122$ (very slow, almost flat)

**Scaling behavior:**
- RMSE with $10k$ samples: ~$11$ points
- RMSE with $600k$ samples: ~$11$ points (no improvement)
- **Conclusion:** Model has high bias (underfitting). It's too simple to capture complex patterns. The remaining ~$11$ point error comes from missing features ($7$ categoricals excluded), non-linear relationships, and irreducible noise.

**Competition context:**
- Best Kaggle score: ~$8.3$ $RMSE$
- My simple model: $11.0$ $RMSE$  
- Gap of only $3$ points despite using $4/12$ features shows this is a reasonable baseline.

**To improve further:**
- Add remaining $7$ categorical features with proper encoding
- Engineer interaction features `study × attendance` and polynomials `sleep²`
- Try ensemble methods `XGBoost`, `LightGBM`

---

## Project Structure

```
kaggle_s6e1_gd/
├── data/
│   └── train.csv
├── 01_gradient_descent_from_scratch.ipynb
├── README.md
└── requirements.txt
```

---

## Setup and Usage

### Prerequisites

- Python 3.9+
- Libraries: NumPy, Pandas, Matplotlib

### Installation

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Download `train.csv` from [Kaggle S6E1](https://www.kaggle.com/competitions/playground-series-s6e1) to `data/` folder

2. Open `01_gradient_descent_from_scratch.ipynb` in Jupyter/VSCode

3. Run all cells sequentially

---

## Key Learnings

### What I Learned

- **Mathematical foundations:** Derived gradient formulas from scratch using chain rule. Understood how

    $\frac{\partial J}{\partial w} = -\frac{1}{N} \sum e_i x_i$ 
 
    tells us exactly how to update weights to reduce loss.

- **Convergence behavior:** Loss decreases rapidly at first ($4280→665$ in $100$ epochs), then flattens as gradients shrink near the minimum. After epoch $200$, improvements become negligible ($<1$ point per epoch).

- **Model capacity matters more than data volume:** Increasing from $10k$ to $600k$ samples didn't improve RMSE (~$11$ points both times). This revealed the model has high bias - it's too simple to capture complex patterns. Need feature engineering, not more data.

- **Weight interpretation:** The learned weights make intuitive sense - `study_hours` weight ($13.07$) is largest, confirming studying is the strongest predictor. `age` weight ($-0.23$) is nearly zero, showing it barely matters.

- **Implementation details:** NumPy broadcasting automatically handles vectorized operations like `(X - mean) / std` across all rows. Understanding array shapes (`X.shape[0]` = samples, `X.shape[1]` = features) is crucial for debugging matrix operations.

### Challenges Faced

**Challenge 1: Understanding the gradient descent update rule**  
Initially confused why `w = w - learning_rate * grad_w` has a minus sign. Solved by working through a concrete example: when underpredicting (positive error), gradient is negative, so `w - (negative) = w + positive` increases the weight, which raises predictions - exactly what we need.

**Challenge 2: Forgetting what variables represent**  
With many arrays (`X`, `y`, `w`, `grad_w`, `errors`, etc.), kept losing track of what each meant. Solved by creating a "Variable Quick Reference" cheat sheet mapping each variable to its shape and meaning (e.g., `w = (4,) weights, one per feature`).


---

## Future Improvements

- [ ] Add remaining 7 categorical features with encoding
- [ ] Implement feature engineering (interactions, polynomials)
- [ ] Compare with sklearn LinearRegression
- [ ] Try regularization (Ridge/Lasso)
- [ ] Implement early stopping
- [ ] Test on Kaggle test set and submit

---

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e1)
- Gradient Descent derivation (personal notes from Session 1-3)
- NumPy Documentation

---

## License

Educational project - free to use and modify.
