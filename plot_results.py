import pandas as pd
import matplotlib.pyplot as plt

# Load test results
results = pd.read_csv("test_results.csv")

# Plot total assets over time
plt.figure(figsize=(10, 6))
plt.plot(results["Step"], results["Total Assets"], label="Total Assets")
plt.title("Agent Performance Over Time")
plt.xlabel("Step")
plt.ylabel("Total Assets")
plt.legend()
plt.grid()
plt.show()

# Plot rewards over time
plt.figure(figsize=(10, 6))
plt.plot(results["Step"], results["Reward"], label="Reward", color="orange")
plt.title("Rewards Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()
