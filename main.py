import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure images directory exists
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# ... (rest of the code remains same until plotting)


# Set seed for reproducibility
np.random.seed(42)

# Generate 400 samples
n_samples = 400
payload = np.random.uniform(1.0, 15.0, n_samples)
concurrency = np.random.uniform(1, 15, n_samples)
X_all = np.column_stack((payload, concurrency))

# Define a realistic boundary with noise
# z = w0*x0 + w1*x1 + b
# Fail if 1.2 * payload + 0.8 * concurrency - 12.0 + noise > 0
z = 1.2 * payload + 0.8 * concurrency - 12.0 + np.random.normal(0, 1.5, n_samples)
y_all = (z > 0).astype(int)

# Split 80/20
indices = np.arange(n_samples)
np.random.shuffle(indices)
train_size = int(0.8 * n_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_test = X_all[test_idx]
y_test = y_all[test_idx]

w = np.zeros(X_train.shape[1])
b = 0


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Prediction Function ---> f_w,b
def prediction(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


# Loss Function
def calculate_loss(X, w, b, y):
    pred = prediction(X, w, b)
    # Clip to avoid log(0) or log(1) which leads to NaN/Inf
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    return -y * np.log(pred) - (1 - y) * np.log((1 - pred))


# Cost Function
def calculate_cost(X, w, b, y):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        cost += calculate_loss(X[i], w, b, y[i])
    return cost / m


def regularized_cost(X, w, b, y, lambda_):
    base_cost = calculate_cost(X, w, b, y)
    m = X.shape[0]
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(w))
    return base_cost + reg_term


# Gradiants calculation
def calculate_gradiant(X, w, b, y):
    m, n = X.shape
    dj_dw = np.zeros(n)  # intial Gradiant
    dj_db = 0  # intial Gradiant
    for i in range(m):
        error = prediction(X[i], w, b) - y[i]
        dj_db += error
        for j in range(n):
            dj_dw += error * X[i][j]
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db


def regularized_gradiant(X, w, b, y, lambda_):
    m = X.shape[0]
    dj_dw, dj_db = calculate_gradiant(X, w, b, y)
    dj_dw += (lambda_ / m) * w
    return dj_dw, dj_db


# Gradient Descent
def gradient_descent(
    X,
    w,
    b,
    y,
    alpha=0.1,
    gradiant_=calculate_gradiant,
    cost_=calculate_cost,
    iter=1000,
    lambda_=0,
):
    n = X.shape[1]
    J_history = []
    for i in range(iter):
        # Check if the gradient function takes lambda_
        try:
            dj_dw, dj_db = gradiant_(X, w, b, y, lambda_=lambda_)
        except TypeError:
            dj_dw, dj_db = gradiant_(X, w, b, y)

        for j in range(n):
            w[j] = w[j] - alpha * dj_dw[j]
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            try:
                cost = cost_(X, w, b, y, lambda_=lambda_)
            except TypeError:
                cost = cost_(X, w, b, y)
            J_history.append(cost)

    return w, b, J_history


# Classification Rule if y >= 0.5 then y = 1
def classify(y):
    m = y.shape[0]
    for i in range(m):
        if y[i] >= 0.5:
            y[i] = 1
        else:
            y[i] = 0
    return y


def get_scaling_params(X):
    """Calculates mean and std for each feature."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Prevent division by zero if std is 0
    std = np.where(std == 0, 1e-15, std)
    return mean, std


def scale_features(X, mean, std):
    """Scales X using provided mean and std (Vectorized)."""
    return (X - mean) / std


def map_feature(X):
    """
    Feature mapping function to polynomial features of degree 2.
    X[:, 0] = x1, X[:, 1] = x2
    Returns: x1, x2, x1^2, x2^2, x1*x2
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    res = np.column_stack((x1, x2, x1**2, x2**2, x1 * x2))
    return res


# training the model with different alphas
alphas = [0.1, 0.01, 0.001]
histories = {}
# get cost history for ploting with different alphas
for a in alphas:
    w_init = np.zeros(X_train.shape[1])
    b_init = 0
    _, _, J_hist = gradient_descent(
        X_train,
        w_init,
        b_init,
        y_train,
        alpha=a,
        gradiant_=calculate_gradiant,
        iter=5000,
    )
    histories[a] = J_hist

# Train definitive model with alpha = 0.01 for other plots
w_initial = np.zeros(X_train.shape[1])
b_initial = 0
w_trained, b_trained, _ = gradient_descent(
    X_train,
    w_initial,
    b_initial,
    y_train,
    alpha=0.01,
    gradiant_=calculate_gradiant,
    iter=5000,
)
y_pred = classify(prediction(X_test, w_trained, b_trained))

accuracy = np.sum(y_pred == y_test) / len(y_test) * 100
print(f"Accuracy with alpha = 0.01: {accuracy}%")

# --- SCALING COMPARISON ---

# 1. Prepare Scaled Data
mu, sigma = get_scaling_params(X_train)
X_train_scaled = scale_features(X_train, mu, sigma)
X_test_scaled = scale_features(X_test, mu, sigma)

# 2. Train model without scaling (Baseline - already done above, but let's be explicit)
# We use alpha=0.01 as a fair baseline for both.
w_unscaled, b_unscaled, J_unscaled = gradient_descent(
    X_train, np.zeros(2), 0, y_train, alpha=0.01, iter=5000
)
y_pred_unscaled = classify(prediction(X_test, w_unscaled, b_unscaled))
acc_unscaled = np.sum(y_pred_unscaled == y_test) / len(y_test) * 100

# 3. Train model with scaling
w_scaled, b_scaled, J_scaled = gradient_descent(
    X_train_scaled, np.zeros(2), 0, y_train, alpha=0.01, iter=5000
)
y_pred_scaled = classify(prediction(X_test_scaled, w_scaled, b_scaled))
acc_scaled = np.sum(y_pred_scaled == y_test) / len(y_test) * 100

print(f"Accuracy (Unscaled): {acc_unscaled:.2f}%")
print(f"Accuracy (Scaled): {acc_scaled:.2f}%")

# 4. Plot Comparison
plt.figure(figsize=(10, 5))

# Subplot 1: Convergence Comparison
plt.subplot(1, 2, 1)
plt.plot(J_unscaled, label="Unscaled", color="red")
plt.plot(J_scaled, label="Scaled", color="green")
plt.title("Convergence (5000 iterations)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Subplot 2: Accuracy Comparison
plt.subplot(1, 2, 2)
bars = plt.bar(
    ["Unscaled", "Scaled"],
    [acc_unscaled, acc_scaled],
    color=["red", "green"],
    alpha=0.7,
)
plt.title("Final Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{yval:.2f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "scaling_comparison.png"))
plt.close()

# --- PLOTTING ---

# 1. Cost History (Learning Curve) - Comparison of Alphas
plt.figure(figsize=(8, 6))
for a in alphas:
    plt.plot(histories[a], label=f"alpha={a}")

plt.title("The Cost History (Learning Curve)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
# Set y-limit to avoid large alpha cost spikes crushing the plot
plt.ylim(0, max(histories[0.01]) * 1.5)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "cost_history.png"))
plt.close()


# Helper function to plot decision boundary
def plot_db(X, y, w, b, title, color="r", label="Decision Boundary", linestyle="--"):
    # Scatter plot
    plt.scatter(
        X[y == 0][:, 0], X[y == 0][:, 1], color="blue", alpha=0.5, label="Succeed (0)"
    )
    plt.scatter(
        X[y == 1][:, 0], X[y == 1][:, 1], color="red", alpha=0.5, label="Fail (1)"
    )

    # Boundary line: w0*x0 + w1*x1 + b = 0 => x1 = (-w0*x0 - b) / w1
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 2
    x0_range = np.linspace(x0_min, x0_max, 100)

    if w[1] != 0:
        x1_boundary = (-w[0] * x0_range - b) / w[1]
        plt.plot(
            x0_range,
            x1_boundary,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Payload Size (MB)")
    plt.ylabel("Concurrency Level")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, X[:, 1].max() + 2)


# 2. Feature Space & Decision Boundary
plt.figure(figsize=(8, 6))
plot_db(X_train, y_train, w_trained, b_trained, "The Feature Space & Decision Boundary")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "decision_boundary.png"))
plt.close()

# --- REGULARIZATION COMPARISON ---

# 1. Choose 3 lambda values to compare
lambdas = [0, 10, 100]
reg_results = {}

print("\n--- Regularization Comparison (on Scaled Data) ---")

for l in lambdas:
    # Use the user's regularized functions
    w_init = np.zeros(X_train_scaled.shape[1])
    b_init = 0
    w_reg, b_reg, J_reg = gradient_descent(
        X_train_scaled,
        w_init,
        b_init,
        y_train,
        alpha=0.01,
        gradiant_=regularized_gradiant,
        cost_=regularized_cost,
        iter=5000,
        lambda_=l,
    )

    y_pred_reg = classify(prediction(X_test_scaled, w_reg, b_reg))
    acc_reg = np.sum(y_pred_reg == y_test) / len(y_test) * 100

    reg_results[l] = {"w": w_reg, "b": b_reg, "acc": acc_reg, "history": J_reg}

    print(
        f"Lambda = {l:3d} | Weights: {w_reg} | Bias: {b_reg:6.4f} | Accuracy: {acc_reg:6.2f}%"
    )

# 2. Plot Regularization Effects
plt.figure(figsize=(15, 5))

# Plot 1: Cost Histories for different lambdas
plt.subplot(1, 2, 1)
for l in lambdas:
    plt.plot(reg_results[l]["history"], label=f"lambda={l}")
plt.title("Regularized Cost History")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Plot 2: Decision Boundary Comparison
# Since we are using scaled data, we need to be careful with plotting.
# We'll plot on the scaled feature space for simplicity in comparing boundaries.
plt.subplot(1, 2, 2)
# Scatter plot of scaled test data
plt.scatter(
    X_test_scaled[y_test == 0][:, 0],
    X_test_scaled[y_test == 0][:, 1],
    color="blue",
    alpha=0.3,
    label="Succeed (0)",
)
plt.scatter(
    X_test_scaled[y_test == 1][:, 0],
    X_test_scaled[y_test == 1][:, 1],
    color="red",
    alpha=0.3,
    label="Fail (1)",
)

x0_range = np.linspace(
    X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5, 100
)
colors = ["green", "orange", "purple"]
for i, l in enumerate(lambdas):
    w_r = reg_results[l]["w"]
    b_r = reg_results[l]["b"]
    if w_r[1] != 0:
        x1_boundary = (-w_r[0] * x0_range - b_r) / w_r[1]
        plt.plot(
            x0_range, x1_boundary, color=colors[i], label=f"lambda={l}", linewidth=2
        )

plt.title("Decision Boundaries (Scaled Space)")
plt.xlabel("Payload Size (Scaled)")
plt.ylabel("Concurrency (Scaled)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "regularization_comparison.png"))
plt.close()

# --- POLYNOMIAL MODEL ---

print("\n--- Polynomial Model Implementation ---")

# 1. Map features to degree 2
X_train_poly = map_feature(X_train)
X_test_poly = map_feature(X_test)

# 2. Scale polynomial features
mu_poly, sigma_poly = get_scaling_params(X_train_poly)
X_train_poly_scaled = scale_features(X_train_poly, mu_poly, sigma_poly)
X_test_poly_scaled = scale_features(X_test_poly, mu_poly, sigma_poly)

# 3. Train polynomial model with regularization
# lambda = 10, alpha = 0.01, iter = 5000
w_poly_init = np.zeros(X_train_poly_scaled.shape[1])
b_poly_init = 0
w_poly, b_poly, J_poly = gradient_descent(
    X_train_poly_scaled,
    w_poly_init,
    b_poly_init,
    y_train,
    alpha=0.01,
    gradiant_=regularized_gradiant,
    cost_=regularized_cost,
    iter=5000,
    lambda_=10,
)

# 4. Accuracy
y_pred_poly = classify(prediction(X_test_poly_scaled, w_poly, b_poly))
acc_poly = np.sum(y_pred_poly == y_test) / len(y_test) * 100
print(f"Polynomial Model Accuracy (lambda=10): {acc_poly:.2f}%")

# 5. Plot Non-Linear Decision Boundary
plt.figure(figsize=(8, 6))

# Scatter original training data
plt.scatter(
    X_train[y_train == 0][:, 0],
    X_train[y_train == 0][:, 1],
    color="blue",
    alpha=0.5,
    label="Succeed (0)",
)
plt.scatter(
    X_train[y_train == 1][:, 0],
    X_train[y_train == 1][:, 1],
    color="red",
    alpha=0.5,
    label="Fail (1)",
)

# Create a mesh grid
x0_min, x0_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x1_min, x1_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx0, xx1 = np.meshgrid(
    np.linspace(x0_min, x0_max, 100), np.linspace(x1_min, x1_max, 100)
)

# Map grid points to polynomial features and scale them
grid_points = np.c_[xx0.ravel(), xx1.ravel()]
grid_poly = map_feature(grid_points)
grid_poly_scaled = scale_features(grid_poly, mu_poly, sigma_poly)

# Predict across the grid
Z = prediction(grid_poly_scaled, w_poly, b_poly)
Z = Z.reshape(xx0.shape)

# Plot the contour at 0.5 (Decision Boundary)
plt.contour(xx0, xx1, Z, levels=[0.5], colors="green", linewidths=3)
# Add a dummy line for the legend
plt.plot([], [], color="green", linewidth=3, label="Poly Decision Boundary")

plt.title("2nd Degree Polynomial Decision Boundary")
plt.xlabel("Payload Size (MB)")
plt.ylabel("Concurrency Level")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "poly_decision_boundary.png"))
plt.close()
