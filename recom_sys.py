# importing necessary libraries
import pandas as pd

# loading movies_metadata
metadata = pd.read_csv('Data/movies_metadata.csv', low_memory=False)
print(metadata.head(3))
