# EXNO6
# AIM:

To Analyze a data set with Various stages of data science.

# ALGORITHM:

Step 1: Choose your own dataset and read it.

Step 2: Include the necessary python Library.

Step 3: Perform Data Preprocessing steps for the necessary columns.

Step 4: Implement Data analysis using the necessary columns.

Step 5: Perform Feature Engineering process for the categorical columns.

Step 6: Implement Advanced data Visualization for the columns necessary.


# CODING AND SCREENSHOTS:
```python
import seaborn as sns
import pandas as pd
df = sns.load_dataset('iris')
print(df.head())
```
![image](https://github.com/user-attachments/assets/608b3ab4-716c-4053-af32-51c0288ffc94)
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Missing values:\n", df.isnull().sum())
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])
print(df.head())
```
![image](https://github.com/user-attachments/assets/1e1147d6-77f0-4b55-af11-029d1d5406b2)
```python
print(df.describe())
correlation_matrix = df.corr()
print("\nCorrelation matrix:\n", correlation_matrix)
sns.pairplot(df, hue='species', palette='viridis')
plt.show()
```
![image](https://github.com/user-attachments/assets/caccd219-c429-4d75-b8b2-2eb1abf835fa)

![image](https://github.com/user-attachments/assets/294695d2-db18-4ee8-9a44-10d550a769b8)
```python
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Feature Correlations")
plt.show()
```
![image](https://github.com/user-attachments/assets/a0f940fa-0017-486b-ad2a-4cf05c92e75f)
```python
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal_length', data=df, palette='muted')
plt.title("Species-wise Distribution of Petal Length")
plt.show()

```
![image](https://github.com/user-attachments/assets/7aad1d0a-cf37-45fa-943c-cf42412ac9b8)

# RESULT:
Thus, the analyzing the dataset with various stages of data science is implemented successfully.
