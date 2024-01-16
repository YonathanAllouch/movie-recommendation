import csv
import pickle
import os
import time
##from openai import OpenAI
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
    openai.api_key = api_key

    embeddings = {}
    request_count = 0

    for title, description in show_descriptions.items():
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=[description] 
            )
            embeddings[title] = response['data'][0]['embedding']
            request_count += 1

            # Add a delay every 3 requests
            if request_count % 3 == 0:
                time.sleep(20)

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
    if os.path.getsize(filepath) == 0:
        logging.error(f"Embeddings file is empty: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        logging.error(f"Embeddings file is corrupted or empty: {filepath}")
        return None    
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
    
    csv_file_path = '/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/imdb_tvshows.csv'  
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
