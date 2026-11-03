from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Data/neo_v2.csv")
df.info()
df.head()