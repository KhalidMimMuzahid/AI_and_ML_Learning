import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 1) Create the data
# Data (units: x=1000 sqft, y=1000s $)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print('x_train =', x_train)
print('y_train =', y_train)

m = x_train.shape[0]
print('m =', m)

# Access i-th example (Python indexing starts at 0)
i = 1
x_i = x_train[i]
y_i = y_train[i]
print(f'(x^({i}), y^({i})) = ({x_i}, {y_i})')


# 2) Plot the points 
plt.scatter(x_train, y_train, marker='x')
plt.title('Housing Prices')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.show()


# 3) Model function (loop version from lab)
def compute_model_output(x, w, b):
    """Compute predictions for all examples using a loop.
    Args:
        x: ndarray shape (m,)
        w, b: scalars
    Returns:
        f_wb: ndarray shape (m,)
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# 4) Vectorized version (recommended) 
def compute_model_output_vec(x, w, b):
    # numpy broadcasting: returns ndarray shape (m,)
    return w * x + b

# 5) Visualize a candidate line (example: w=100, b=100)
w = 100
b = 100
pred = compute_model_output_vec(x_train, w, b)

plt.plot(x_train, pred, label=f'Prediction w={w}, b={b}')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual')
plt.legend()
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s $)')
plt.show()

### 6) Compute the exact line that fits the two points (analytic solution)
# Because we have two points, there's a unique straight line passing through them. Use slope/intercept formulas:

# Points (x1,y1) and (x2,y2)
x1, y1 = x_train[0], y_train[0]
x2, y2 = x_train[1], y_train[1]

# slope w and intercept b
w_analytic = (y2 - y1) / (x2 - x1)
b_analytic = y1 - w_analytic * x1
print('Analytic w =', w_analytic)
print('Analytic b =', b_analytic)

# Plot analytic line
pred_analytic = compute_model_output_vec(x_train, w_analytic, b_analytic)
plt.plot(x_train, pred_analytic, label=f'Analytic fit w={w_analytic}, b={b_analytic}')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.legend()
plt.show()
# Expected analytic values for this dataset: `w = 200`, `b = 100`.

# 7) Use the model to predict 1200 sqft (x = 1.2)
w, b = 200, 100
x_new = 1.2
pred_new = w * x_new + b
print(f'Predicted price for 1200 sqft: ${pred_new:.0f} thousand dollars')
# => $340 thousand dollars