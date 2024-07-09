import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_movie=pd.read_csv("/content/Movie dataset.csv",encoding=("ISO-8859-1"),sep=",",engine='python')
df_movie.dropna(inplace=True)
df_movie.head()
df_movie.shape
df_movie.describe()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df_movie['Genre']= labelencoder.fit_transform(df_movie['Genre'])

df_movie.head()
df_movie.isna().sum()
df2=df_movie.drop(['Votes'],axis=1)
df2.head()
df2.describe()
df2.isna().sum()
df_final=df2.dropna()
df_final.shape   
df_final.head()
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn plots
sns.set(style="whitegrid")

# Plotting a histogram of the 'Rating' column
plt.figure(figsize=(10, 6))
sns.histplot(df2['Rating'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Plotting a count plot of the 'Genre' column
plt.figure(figsize=(10, 6))
sns.countplot(y='Genre', data=df2, order=df2['Genre'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Plotting a bar plot of the top 10 Directors
plt.figure(figsize=(10, 6))
top_directors = df2['Director'].value_counts().nlargest(10)
sns.barplot(x=top_directors.values, y=top_directors.index, palette='rocket')
plt.title('Top 10 Directors')
plt.xlabel('Count')
plt.ylabel('Director')
plt.show()
