
import pandas as pd
import random
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

random.seed(42)

# DEMO DATA ONLY — replace with real/public data
# before reporting final hackathon results.
rows = []

for _ in range(500):
    cgpa = round(random.uniform(4, 10), 2)
    attendance = random.randint(40, 100)
    aptitude = random.randint(20, 100)
    coding = random.randint(10, 100)
    communication = random.randint(20, 100)
    internship = random.randint(0, 1)

    # Illustrative rule for creating synthetic labels.
    score = (
        cgpa * 5
        + attendance * 0.10
        + aptitude * 0.12
        + coding * 0.12
        + communication * 0.08
        + internship * 5
    )

    # Add randomness so the demo labels aren't deterministic.
    score += random.gauss(0, 5)
    placed = int(score >= 65)

    rows.append({
        "CGPA": cgpa,
        "Attendance": attendance,
        "Aptitude Score": aptitude,
        "Coding Score": coding,
        "Communication Score": communication,
        "Internship Experience": internship,
        "Placement Status": placed
    })

df = pd.DataFrame(rows)

# Save practice data.
df.to_csv("students.csv", index=False)

features = [
    "CGPA",
    "Attendance",
    "Aptitude Score",
    "Coding Score",
    "Communication Score",
    "Internship Experience"
]

X = df[features]
y = df["Placement Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("DEMO MODEL TRAINED SUCCESSFULLY!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Demo accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")
print("\nDemo classification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Save model and feature order for the dashboard.
with open("placement_model.pkl", "wb") as file:
    pickle.dump({"model": model, "features": features}, file)

print("\nSaved: students.csv")
print("Saved: placement_model.pkl")
print("NOTE: Results are from synthetic data, not real placements.")
