import pandas as pd 

'''
    Simple Score Based Filtering:

    A simple recommender that filters and scores values based on the ratings and then lists the best ones.
'''
# Load Movies Metadata
metadata = pd.read_csv('../data/movies_metadata.csv', low_memory=False)

# Mean of average vote column
C = metadata['vote_average'].mean() 

# Minimum number of votes needed to be qualified
m = metadata['vote_count'].quantile(0.90)

# Movies that qualified
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

# Weighted ratings for every movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)

# Introducing score as new attribtue in dataframe
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

'''
    Content Based Filtering:

    For the next section, gonna try and build a content-based filter. Firstly, trying to recommend movies that are similar in terms of plot.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Define vectorizer and remove stop words like 'an', 'the', etc
tfidf = TfidfVectorizer(stop_words='english')

# NaN replaced with empty string
metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Fit NearestNeighbors model using cosine distance
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# Create a reverse mapping from movie titles to DataFrame indices
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Recommendation function using NearestNeighbors
def similarContent(title, n=15):
    idx = indices[title]
    
    # Find top n+1 neighbors (including the movie itself)
    distances, neighbors = model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    
    # Skip the first result since it's the movie itself
    similar_indices = neighbors[0][1:]
    
    return metadata['title'].iloc[similar_indices]

'''
    Director, Genre, and Keywords Based Filtering:

    For the last section, gonna use various parameters to influence the recommendations.
'''

credits = pd.read_csv('../data/credits.csv')
keywords = pd.read_csv('../data/keywords.csv')

# Remove rows with bad IDs
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to integers so that can be merged
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge credits and keywords into main DF
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# parse stringified features into respective objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

import numpy as np 

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_top_3(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]

        if len(names) > 3:
            names = names[:3]
        return names

    return [] # if missing or incorrect data

metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(get_top_3)

print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))