# Linear Regression with Gradient Descent and Visualization
# ----------------------------------------------------------
# Author: Khalid Mim Muzahid
# Description:
# This script demonstrates gradient descent for linear regression
# with cost surface visualization, cost reduction plot, and fitted line.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------
# 1️⃣ Generate synthetic data
# ------------------------------------------
np.random.seed(42)
X = np.linspace(0, 2000, 100)
y = 100 + 0.1 * X + np.random.randn(100) * 40  # Linear relation + noise

# ------------------------------------------
# 2️⃣ Define helper functions
# ------------------------------------------

def compute_cost(X, y, w, b):
    """Compute mean squared error cost."""
    m = len(y)
    cost = (1 / (2 * m)) * np.sum((w * X + b - y) ** 2)
    return cost

def compute_gradient(X, y, w, b):
    """Compute gradient for parameters w and b."""
    m = len(y)
    dj_dw = (1 / m) * np.sum((w * X + b - y) * X)
    dj_db = (1 / m) * np.sum(w * X + b - y)
    return dj_dw, dj_db

def gradient_descent(X, y, w_init, b_init, alpha, iterations):
    """Perform gradient descent and record cost history."""
    w, b = w_init, b_init
    cost_history = []
    params_history = []
    
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        params_history.append((w, b))

        if i % 50 == 0:
            print(f"Iteration {i:4d}: Cost={cost:.2f}, w={w:.4f}, b={b:.4f}")

    return w, b, cost_history, params_history

# ------------------------------------------
# 3️⃣ Initialize parameters
# ------------------------------------------
w_init = -0.1
b_init = 900
alpha = 0.0000005
iterations = 500

# ------------------------------------------
# 4️⃣ Run Gradient Descent
# ------------------------------------------
w_final, b_final, cost_history, params_history = gradient_descent(X, y, w_init, b_init, alpha, iterations)
print("\n✅ Final parameters:")
print(f"w = {w_final:.4f}, b = {b_final:.4f}")

# ------------------------------------------
# 5️⃣ Visualization 1: Data and fitted line
# ------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Training Data", alpha=0.6)
plt.plot(X, w_final * X + b_final, color='red', label='Fitted Line')
plt.title("Linear Regression Fit")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------
# 6️⃣ Visualization 2: Cost Reduction over Iterations
# ------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(cost_history, color='purple')
plt.title("Cost Reduction over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

# ------------------------------------------
# 7️⃣ Visualization 3: Cost Function Contour and Surface
# ------------------------------------------
w_range = np.linspace(-0.2, 0.3, 50)
b_range = np.linspace(0, 200, 50)
W, B = np.meshgrid(w_range, b_range)

J = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J[i, j] = compute_cost(X, y, W[i, j], B[i, j])

# --- Contour Plot ---
plt.figure(figsize=(8, 6))
plt.contour(W, B, J, levels=30, cmap='viridis')
plt.scatter([w_init], [b_init], color='red', label='Start (Init)')
plt.scatter([w_final], [b_final], color='orange', label='End (Final)')
plt.title("Cost Function Contour")
plt.xlabel("w")
plt.ylabel("b")
plt.legend()
plt.grid(True)
plt.show()

# --- 3D Surface Plot ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, J, cmap='viridis', alpha=0.8)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Cost")
ax.set_title("Cost Function Surface")
plt.show()

# ------------------------------------------
# 8️⃣ Prediction Example
# ------------------------------------------
test_size = 1250
predicted_price = w_final * test_size + b_final
print(f"\n🏠 Predicted price for a 1250 sq ft house: ${predicted_price:.2f}")
