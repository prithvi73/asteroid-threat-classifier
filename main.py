from sklearn.model_selection import train_test_split
import pandas as pd

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
