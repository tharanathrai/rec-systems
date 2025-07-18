import pandas as pd 

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

# So far, this was a simple recommender that filtered and scored values based on the ratings and then listed the best ones

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

print(similarContent('Dilwale Dulhania Le Jayenge'))