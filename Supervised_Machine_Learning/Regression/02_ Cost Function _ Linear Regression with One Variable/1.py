# ======================================================================
# Linear Regression with One Variable - Cost Function (Standalone Version)
# ======================================================================

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
# %matplotlib inline

# -------------------------------------------------------------
# 1️⃣ Problem Statement
# -------------------------------------------------------------
# You want to predict housing prices based on house size.
# Dataset: (size in 1000 sqft) vs (price in 1000s of dollars)

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# -------------------------------------------------------------
# 2️⃣ Model Function
# -------------------------------------------------------------
# f_wb = w * x + b

def predict(x, w, b):
    return w * x + b

# Test the model function
print("\nPrediction example:")
print(f"f_wb for x=1.0, w=200, b=100 → {predict(1.0, 200, 100)} (expected 300)")

# -------------------------------------------------------------
# 3️⃣ Cost Function
# -------------------------------------------------------------
# J(w, b) = (1 / (2m)) * Σ (f_wb(x_i) - y_i)^2

def compute_cost(x, y, w, b): 
    m = x.shape[0] 
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# -------------------------------------------------------------
# 4️⃣ Try different values for w and b
# -------------------------------------------------------------
test_w = 200
test_b = 100
cost = compute_cost(x_train, y_train, test_w, test_b)

print(f"\nCost at w={test_w}, b={test_b}: {cost:.2f} (expected 0.00 for perfect fit)")

# Try another
cost = compute_cost(x_train, y_train, 150, 100)
print(f"Cost at w=150, b=100: {cost:.2f}")

# -------------------------------------------------------------
# 5️⃣ Visualizing Cost vs w
# -------------------------------------------------------------
# We'll fix b = 100 and vary w

w_values = np.linspace(0, 400, 100)
cost_values = [compute_cost(x_train, y_train, w, 100) for w in w_values]

plt.figure(figsize=(8, 5))
plt.plot(w_values, cost_values, color='royalblue')
plt.title("Cost vs Weight (w) for b=100")
plt.xlabel("w (slope)")
plt.ylabel("Cost J(w, b)")
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# 6️⃣ Visualize the data and the best-fit line
# -------------------------------------------------------------
best_w, best_b = 200, 100

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color='red', label='Training data')
plt.plot(x_train, predict(x_train, best_w, best_b), color='blue', label=f'Prediction line (w={best_w}, b={best_b})')
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price ($1000s)")
plt.title("Housing Prices vs Size")
plt.legend()
plt.show()

# -------------------------------------------------------------
# 7️⃣ Larger Data Set Example
# -------------------------------------------------------------
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

def plot_dataset(x, y):
    plt.scatter(x, y, color='red')
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price ($1000s)")
    plt.title("Larger Housing Dataset")
    plt.show()

plot_dataset(x_train, y_train)

# Try a few w, b combinations
c1 = compute_cost(x_train, y_train, 200, 100)
c2 = compute_cost(x_train, y_train, 210, 2.4)
print(f"Cost for w=200, b=100: {c1:.2f}")
print(f"Cost for w=210, b=2.4: {c2:.2f}")

# -------------------------------------------------------------
# 8️⃣ 3D Visualization of Cost Function
# -------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

w_vals = np.linspace(0, 300, 100)
b_vals = np.linspace(-100, 200, 100)
J_vals = np.zeros((len(w_vals), len(b_vals)))

for i in range(len(w_vals)):
    for j in range(len(b_vals)):
        J_vals[i, j] = compute_cost(x_train, y_train, w_vals[i], b_vals[j])

W, B = np.meshgrid(w_vals, b_vals)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, J_vals.T, cmap='viridis', alpha=0.8)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost')
ax.set_title("Cost Function Surface (J vs w, b)")
plt.show()

