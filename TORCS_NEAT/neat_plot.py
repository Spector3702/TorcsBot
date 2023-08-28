import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

FOLDER_NAME = 'TORCS_NEAT'
# Load rewards from the saved file
with open(f'{FOLDER_NAME}/reward/neat_rewards.pkl', 'rb') as f:
    all_rewards = pickle.load(f)

print(len(all_rewards))
# Calculate the average reward for each group of 10 generations
l = len(all_rewards) // 10
avg = []
all_rewards_alt = [item for sublist in all_rewards for item in sublist]
for i in range(l):
    gen = all_rewards_alt[i * 10: (i + 1) * 10]
    total = sum(_ for _ in gen)
    avg.append(total / 10)

plt.figure(figsize=(16, 8))  # Adjust figure size if needed
plt.subplot(1, 2, 1)  # Subplot: 1 row, 2 columns, plot 1
print(avg)
# Plot the average rewards with generations as x-axis
plt.plot(avg, marker='o')
plt.xlabel('Generations')
plt.ylabel('Average Reward')
plt.title('Average Reward per Generation')
# Set integer ticks for x-axis
x_ticks = list(range(0, len(avg) * 1, 5))
plt.xticks(x_ticks, x_ticks)
plt.grid(True)

plt.subplot(1, 2, 2)  # Subplot: 1 row, 2 columns, plot 2
# ------------------------------------------------------------------------- func. switchiing part
# Convert the x-axis values to an array of integers (0, 1, 2, ...)
x = np.arange(len(all_rewards_alt)).reshape(-1, 1)
# Fit a linear regression model
model = LinearRegression() # Linear
model.fit(x, all_rewards_alt)
y_pred = model.predict(x)

# Create the scatter plot
plt.scatter(x, all_rewards_alt, marker='.', color='orange', label='Data points')
plt.plot(x, y_pred, color='#6688FF', linestyle='dashed', label='Trendline')
# -------------------------------------------------------------------------
plt.xlabel('Genomes')
plt.ylabel('Reward')
plt.title('Reward per Genome')

plt.legend()
plt.show()

"""
Different Types of Trendline : 
--------------------------------------------------------------------------------
linear
--------------------------------------------------------------------------------
# Convert the x-axis values to an array of integers (0, 1, 2, ...)
x = np.arange(len(all_rewards_alt)).reshape(-1, 1)
# Fit a linear regression model
model = LinearRegression() # Linear
model.fit(x, all_rewards_alt)
y_pred = model.predict(x)

# Create the scatter plot
plt.scatter(x, all_rewards_alt, marker='.', color='orange', label='Data points')
plt.plot(x, y_pred, color='#6688FF', linestyle='dashed', label='Trendline')
--------------------------------------------------------------------------------
polynomial
--------------------------------------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(len(all_rewards_alt)).reshape(-1, 1)

# Fit a polynomial regression model
degree = 2  # Choose the degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, all_rewards_alt)
y_pred = model.predict(x_poly)

# Create the scatter plot
plt.scatter(x, all_rewards_alt, marker='o', color='orange', label='Data points')
plt.plot(x, y_pred, color='#6688FF', linestyle='dashed', label=f'Degree {degree} Polynomial Fit')

"""