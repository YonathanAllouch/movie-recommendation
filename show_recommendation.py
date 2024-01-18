import pickle
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
    print("Recommended TV Shows:")
    for show, score in recommendations.items():
        print(f"{show} ({score}%)")
    return recommendations

