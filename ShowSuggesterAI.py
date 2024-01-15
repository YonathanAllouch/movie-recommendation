import csv
import pickle
import os
import openai
import logging
from dotenv import load_dotenv

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tv_shows(csv_file):
    if not os.path.exists(csv_file):
        logging.error(f"CSV file not found: {csv_file}")
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    shows = {}
    try:
        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                shows[row['Title']] = row['Description']
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise
    return shows
       

def get_embeddings(show_descriptions, api_key):
    if not api_key:
        raise ValueError("API key is missing")
    
    openai.api_key = api_key
    embeddings = {}
    for title, description in show_descriptions.items():
        try:
            response = openai.Embedding.create(
                input=description,
                model="text-similarity-babbage-001"  # You can choose a different model if needed
            )
            embeddings[title] = response['data'][0]['embedding']
        except Exception as e:
            logging.error(f"Error fetching embeddings for {title}: {e}")
            raise    

    return embeddings

def save_embeddings(embeddings, filepath):
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(embeddings, file)
    except Exception as e:
        logging.error(f"Error saving embeddings to file: {e}")
        raise        

def load_embeddings(filepath):
    if not os.path.exists(filepath):
        logging.info(f"Embeddings file not found: {filepath}")
        return None

    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading embeddings from file: {e}")
        raise




def main():
    # File paths
    load_dotenv()
    api_key = os.getenv('API_KEY')  
    if not api_key:
        logging.error("API Key not found. Please set API_KEY in your environment.")
        return
    
    csv_file_path = '/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/imdb_tvshows.csv'  # Replace with your actual CSV file path
    pickle_file_path = 'embeddings.pkl'

    # Try to load embeddings from the pickle file
    try:
        embeddings = load_embeddings(pickle_file_path)
        # If embeddings are not found, read the TV shows and get embeddings
        if embeddings is None:
            shows = load_tv_shows(csv_file_path)
            embeddings = get_embeddings(shows, api_key)
            save_embeddings(embeddings, pickle_file_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

        # Your further processing with embeddings can go here
        # ...

if __name__ == "__main__":
    main()
