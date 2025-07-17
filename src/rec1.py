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

#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

# So far, this was a simple recommender that filtered and scored values based on the ratings and then listed the best ones