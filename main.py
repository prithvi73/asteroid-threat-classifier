from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Reading the data
df = pd.read_csv("Data/neo_v2.csv")
# df.info()
# df.head()

# Drop missing values
df = df.dropna()

# Drop unnecessary columns
df = df.drop(columns=["name", "orbiting_body", "id"])

# Encoding the booleans into integers
df["hazardous"] = df["hazardous"].astype(int)
df["sentry_object"] = df["sentry_object"].astype(int)

# df.info()

# Isoalting Features and Target Matrix
X = df[["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "sentry_object", "absolute_magnitude"]]
y = df["hazardous"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Testing the model
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

# Testing the model on data not in our testing data
mystery_asteroid = pd.DataFrame([[1.5, 3.5, 85000.5, 1200000.0, 0, 15.2]], columns=X.columns)
mystery_pred = clf.predict(mystery_asteroid)
print(f"Threat detected (1=Hazardous, 0=Safe) : {mystery_pred}")