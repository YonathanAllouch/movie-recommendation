import csv
import pickle
import os
import openai
from dotenv import load_dotenv


def load_tv_shows(csv_file):
    shows = {}
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            shows[row['Title']] = row['Description']
    return shows

def get_embeddings(show_descriptions, api_key):
    openai.api_key = api_key

    embeddings = {}
    for title, description in show_descriptions.items():
        response = openai.Embedding.create(
            input=description,
            model="text-similarity-babbage-001"  # You can choose a different model if needed
        )
        embeddings[title] = response['data'][0]['embedding']

    return embeddings

def save_embeddings(embeddings, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(embeddings, file)

def load_embeddings(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    return None

import os



def main():
    # File paths
    csv_file_path = '/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/imdb_tvshows.csv'  # Replace with your actual CSV file path
    pickle_file_path = 'embeddings.pkl'
    api_key = os.getenv('API_KEY')  

    # Try to load embeddings from the pickle file
    embeddings = load_embeddings(pickle_file_path)

    # If embeddings are not found, read the TV shows and get embeddings
    if embeddings is None:
        shows = load_tv_shows(csv_file_path)
        embeddings = get_embeddings(shows, api_key)

        # Save embeddings for future use
        save_embeddings(embeddings, pickle_file_path)

    # Your further processing with embeddings can go here
    # ...

if __name__ == "__main__":
    main()
