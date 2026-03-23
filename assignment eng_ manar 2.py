import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

data = {
    'Age': [25, np.nan, 30, 22, np.nan],
    'Salary': [5000, 6000, np.nan, 4500, 5200],
    'City': ['Cairo', 'Alex', 'Cairo', 'Giza', 'Alex']
}

df = pd.DataFrame(data)

cat_col = 'City'

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[[cat_col]])
encoded_cols = encoder.get_feature_names_out([cat_col])
encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

idx = df.columns.get_loc(cat_col)
df = df.drop(columns=[cat_col])

for i, col in enumerate(encoded_cols):
    df.insert(idx + i, col, encoded_df[col])

missing_percent = (df.isnull().sum() / len(df)) * 100

low = missing_percent[missing_percent <= 30].index
high = missing_percent[missing_percent > 30].index

if len(low) > 0:
    df[low] = SimpleImputer(strategy='mean').fit_transform(df[low])

if len(high) > 0:
    df[high] = KNNImputer(n_neighbors=2).fit_transform(df[high])

print(df)