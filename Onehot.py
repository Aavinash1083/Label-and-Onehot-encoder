import numpy as np
import pandas as pd

dataset = pd.read_csv("sample.csv")
print(dataset)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

dataset = np.array(columnTransformer.fit_transform(dataset), dtype = np.str)
print(dataset)



