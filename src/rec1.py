import pandas as pd 

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

C = metadata['vote_average'].mean()

m = metadata['vote_count'].quantile(0.90)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]