import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the independent variables
X1 = np.linspace(-5, 5, 100)  # Feature 1
X2 = np.linspace(-5, 5, 100)  # Feature 2
X1, X2 = np.meshgrid(X1, X2)

# Coefficients for the linear combination (weights)
beta_0 = 1  # Intercept
beta_1 = 0.5  # Coefficient for X1
beta_2 = 0.3  # Coefficient for X2

# Calculate the linear combination
Z = beta_0 + beta_1 * X1 + beta_2 * X2

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Labels and title
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')
ax.set_zlabel('Linear Combination (Z)')
ax.set_title('Linear Combination of Inputs in Logistic Regression')

# Show the plot
plt.show()