import os
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import numpy as np

def sanitize_filename(filename):
    # Remove special characters from the filename
    return re.sub(r'[<>:"/\\|?*]', '', filename)
# Get the current working directory
current_directory = os.getcwd()
# File paths for reading the CSV file and saving the plots
csv_file_path = os.path.join(current_directory, "DESCRIPTOR.csv")
plots_folder = os.path.join(current_directory, "shap_plots")
# Create the folder for saving plots if it doesn't exist
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
# Load the dataset
dataset = pd.read_csv(csv_file_path)
X = dataset.drop('Column0', axis=1)
y = dataset['Column0']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit the Random Forest model for regression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#######################
# Create the SHAP explainer using the model itself as the model function
explainer = shap.Explainer(model)
# Calculate SHAP values for the test set
sv = explainer.shap_values(X_test)
idx = 3
# Create an Explanation object for the single prediction
exp = shap.Explanation(sv[idx], model.predict(X_test)[idx], X_test.iloc[idx])
# Plot the waterfall plot
shap.plots.waterfall(exp)
plt.savefig(os.path.join(plots_folder, "Waterfall_plot.png"))
plt.close()
# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)
# Get the top 3 features based on their mean absolute SHAP values
top_3_features = pd.DataFrame(shap_values).abs().mean().nlargest(3).index
# Plot summary plot and save it
summary_plot = shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, plot_type="bar", show=False)
plt.savefig(os.path.join(plots_folder, "shap_summary_plot.png"))

# Plot force plot for a single prediction and save it
index_to_explain = 0  # You can change this index to explain different test samples
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[index_to_explain], X_test.iloc[index_to_explain], feature_names=X_test.columns, show=False)
shap.save_html(os.path.join(plots_folder, "shap_force_plot.html"), force_plot)
plt.close()
# Plot dependence plot for each of the top 3 features and save it
for feature_to_plot in top_3_features:
    dependence_plot = shap.dependence_plot(feature_to_plot, shap_values, X_test, feature_names=X_test.columns, show=False)
    filename = f"shap_dependence_plot_{feature_to_plot}.png"
    plt.savefig(os.path.join(plots_folder, sanitize_filename(filename)))
    plt.close()
# Plot interaction values between the top 3 feature pairs and save them
for i in range(len(top_3_features)):
    for j in range(i+1, len(top_3_features)):
        feature_interaction1 = top_3_features[i]
        feature_interaction2 = top_3_features[j]
        interaction_plot = shap.dependence_plot(feature_interaction1, shap_values, X_test, feature_names=X_test.columns, interaction_index=feature_interaction2, show=False)
        filename = f"shap_interaction_plot_{feature_interaction1}_vs_{feature_interaction2}.png"
        plt.savefig(os.path.join(plots_folder, sanitize_filename(filename)))
        plt.close()
# Plot interaction summary plot and save it
interaction_summary_plot = shap.summary_plot(shap_values, X_test, plot_type='interaction', show=False)
plt.savefig(os.path.join(plots_folder, "shap_interaction_summary_plot.png"))
plt.close()
'''
# Plot scatter plot for two features and save it
scatter_plot = shap.plots.scatter(shap_values[:, 0], color=X_test.iloc[:, 0], show=False)
plt.savefig(os.path.join(plots_folder, "shap_scatter_plot.png"))
plt.close()
# Plot decision plot for a single prediction and save it
decision_plot = shap.decision_plot(explainer.expected_value, shap_values[index_to_explain], feature_names=X_test.columns, show=False)
plt.savefig(os.path.join(plots_folder, "shap_decision_plot.png"))
plt.close()
# Plot dependence heatmap and save it
dependence_heatmap = shap.plots.heatmap(shap_values[0], show=False)
plt.savefig(os.path.join(plots_folder, "shap_dependence_heatmap.png"))
plt.close()
# Plot dependence matrix and save it
dependence_matrix = shap.plots.scatter(shap_values, max_display=3, show=False)
plt.savefig(os.path.join(plots_folder, "shap_dependence_matrix.png"))
plt.close()'''






