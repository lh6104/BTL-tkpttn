import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.power import FTestAnovaPower
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
data_path = r"C:\\Users\\Laptop K1\\OneDrive\\Desktop\\ys1a.csv"
df = pd.read_csv(data_path)

# Keep only relevant columns for the analysis
selected_columns = ['ys', 'vec', 'deltachi', 'delta', 'deltahmix', 'deltasmix']
df = df[selected_columns]

# Convert columns to numeric and handle missing data
for col in df.columns:
    df[col] = df[col].replace({',': ''}, regex=True)  # Remove commas
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric

# Handle NaN values
# Replace NaN in numeric columns with the column mean
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())


# Display the cleaned dataset for verification
print("Cleaned data (first 5 rows):")
print(df.head())


# Define the target variable (ys) and input features
target_column = 'ys'
features = ['vec', 'deltachi', 'delta', 'deltahmix', 'deltasmix']

X = df[features]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Define parameter grid for SVR
C_values = [100, 200]
epsilon_values = [0.1, 0.5]

# Estimate the required number of replications
alpha = 0.05  # Significance level
power = 0.8   # Desired power
effect_size = 0.5  # Assumed medium effect size (Cohen's f)

power_analysis = FTestAnovaPower()
required_replications = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=len(C_values) * len(epsilon_values))
required_replications = int(np.ceil(required_replications))

print(f"Estimated number of replications: {required_replications}")

# Prepare a list to store results
results = []

# Perform experiments with SVR
for C in C_values:
    for epsilon in epsilon_values:
        for replicate in range(required_replications):
            svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
            svr.fit(X_train, y_train)

            # Predict on the test set
            y_pred = svr.predict(X_test)

            # Calculate RMSE and MAPE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Calculate mean and standard deviation of ys for the replicate
            ys_mean = y_train.mean()
            ys_std = y_train.std()

            # Store results
            results.append({
                'C_value': C,
                'epsilon': epsilon,
                'replicate': replicate,
                'RMSE': rmse,
                'MAPE': mape,
                'ys_mean': ys_mean,
                'ys_std': ys_std
            })

# Convert results into a DataFrame
results_df = pd.DataFrame(results)


# Display first few rows of the results for verification
print("Sample results with ys statistics:")
print(results_df.head())


# Save full results table for further inspection
results_df.to_csv("full_results.csv", index=False)
print("Full results saved to 'full_results.csv'")

# Perform repeated measures ANOVA for RMSE
anova_rmse = AnovaRM(data=results_df, depvar='RMSE', subject='replicate', within=['C_value', 'epsilon'])
anova_results_rmse = anova_rmse.fit()

# Display ANOVA results for RMSE
print("ANOVA Results for RMSE")
print(anova_results_rmse)

# Perform repeated measures ANOVA for MAPE
anova_mape = AnovaRM(data=results_df, depvar='MAPE', subject='replicate', within=['C_value', 'epsilon'])
anova_results_mape = anova_mape.fit()

# Display ANOVA results for MAPE
print("ANOVA Results for MAPE")
print(anova_results_mape)

# Plot RMSE and MAPE for better visualization
plt.figure(figsize=(10, 5))

# Plot RMSE
plt.subplot(1, 2, 1)
results_df.groupby(['C_value', 'epsilon'])['RMSE'].mean().unstack().plot(kind='bar', ax=plt.gca())
plt.title('Average RMSE by Parameters')
plt.ylabel('RMSE')
plt.xlabel('C_value')
plt.legend(title='Epsilon')

# Plot
plt.subplot(1, 2, 2)
results_df.groupby(['C_value', 'epsilon'])['MAPE'].mean().unstack().plot(kind='bar', ax=plt.gca())
plt.title('Average MAPE by Parameters')
plt.ylabel('MAPE')
plt.xlabel('C_value')
plt.legend(title='Epsilon')

plt.tight_layout()
plt.show()
