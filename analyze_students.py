
import pandas as pd
import pickle

# Load the trained demo model
with open("placement_model.pkl", "rb") as file:
    saved = pickle.load(file)

model = saved["model"]
features = saved["features"]

# Load the sample student dataset
df = pd.read_csv("students.csv")

# Prepare student inputs
X = df[features]

# Predict placement outcomes
df["Predicted Placement"] = model.predict(X)
df["Placement Probability (%)"] = (
    model.predict_proba(X)[:, 1] * 100
).round(1)

# Identify students who may need extra support
df["Support Recommendation"] = df.apply(
    lambda row: (
        "Review skills and offer support"
        if row["Predicted Placement"] == 0
        else "Continue placement preparation"
    ),
    axis=1
)

# Save the analysis report
df.to_csv("placement_analysis.csv", index=False)

print("Student analysis completed!")
print("Total students analyzed:", len(df))
print("\nPredicted placement outcomes:")
print(df["Predicted Placement"].value_counts())
print("\nStudents flagged for additional support:")
print((df["Predicted Placement"] == 0).sum())
print("\nReport saved as placement_analysis.csv")
