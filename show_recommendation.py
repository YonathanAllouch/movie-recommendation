import pickle
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from closest_match_show import match_shows, get_favorite_tv_shows

def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def calculate_average_vector(shows, embeddings):
    vectors = [embeddings[show] for show in shows]
    return np.mean(vectors, axis=0)

def recommend_shows(user_shows, all_shows, embeddings):
    avg_vector = calculate_average_vector(user_shows, embeddings)
    similarities = {show: cosine_similarity([avg_vector], [embeddings[show]])[0][0] 
                    for show in all_shows if show not in user_shows}
    sorted_shows = sorted(similarities, key=similarities.get, reverse=True)[:5]
    top_shows = sorted_shows[:min(5, len(sorted_shows))]
    max_similarity = max(similarities.values(),default=1)
    recommendations = {show: round((similarities[show] / max_similarity) * 100) 
                       for show in top_shows}
    return recommendations

# Load embeddings
embeddings = load_embeddings('embeddings.pkl')
'''
# Get user input
known_shows = list(embeddings.keys())
user_shows = get_favorite_tv_shows(known_shows)
recommended_shows = recommend_shows(user_shows, known_shows, embeddings)

for show, score in recommended_shows.items():
    print(f"{show} ({score}%)")
'''