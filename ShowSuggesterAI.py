import csv
import re
import pickle
import os
from openai import OpenAI
import logging
from dotenv import load_dotenv
from closest_match_show import get_favorite_tv_shows
from show_recommendation import recommend_shows
from generate_TV_show import generate_show_description, generate_show_image, save_and_open_image , describe_picture_for_story    

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
       

def get_embeddings(show_descriptions, batch_size=200):

    embeddings = {}
    descriptions_batch = []
    titles_batch = []

    for title, description in show_descriptions.items():
        descriptions_batch.append(description)
        titles_batch.append(title)
        # Once batch size is reached, or end of descriptions is reached, make the API call
    
    if len(descriptions_batch) == batch_size or title == list(show_descriptions.keys())[-1]:
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=descriptions_batch
            )
            for i, embedding in enumerate(response['data']):
                embeddings[titles_batch[i]] = embedding['embedding']
                
                # Reset the batches
            descriptions_batch = []
            titles_batch = []

                # Wait 60 seconds to respect the rate limit
                #time.sleep(60)

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

def parse_show_details(text):
    # Regex pattern to match "title : description"
    pattern = r'(.+?)\s*:\s*(.+)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Check if match is found
    if match:
        title = match.group(1).strip() + '"'
        description = '"' + match.group(2).strip()
        return title, description
    else:
        # Return a default value or handle the error as needed
        return None, None


# Load the API key from the environment variable
try: 
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"An error occurred: {e}")
    raise   

def main():
    # File paths
    load_dotenv()    
    csv_file_path = '/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/imdb_tvshows.csv'  
    pickle_file_path = 'embeddings.pkl'

    # Try to load embeddings from the pickle file
    try:
        embeddings = load_embeddings(pickle_file_path)
        # If embeddings are not found, read the TV shows and get embeddings
        if embeddings is None:
            shows = load_tv_shows(csv_file_path)
            embeddings = get_embeddings(shows, client)
            save_embeddings(embeddings, pickle_file_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")     

    # Get user input
    known_shows = list(embeddings.keys())
     # Step 1: Get user's favorite shows
    favorite_shows = get_favorite_tv_shows(known_shows)

    # Step 2: Generate recommendations
    recommended_shows = recommend_shows(favorite_shows, known_shows, embeddings)

    # Steps 3 & 4: Generate and parse descriptions for new shows
    description_show_create = generate_show_description("user's favorite", favorite_shows)
    recommended_based_show = generate_show_description("my recommendations", recommended_shows)
    title1, description1 = parse_show_details(description_show_create)
    title2, description2 = parse_show_details(recommended_based_show)

    convert_description_to_picture1 = describe_picture_for_story(description_show_create)
    convert_description_to_picture2 = describe_picture_for_story(recommended_based_show)
    # Step 5: Generate images for new shows
    image_show_create_url = generate_show_image(convert_description_to_picture1)
    recommended_show_image_url = generate_show_image(convert_description_to_picture2)

    # Step 6: Display show details and save images
    print(f"I have also created just for you two shows which I think you would love. "
        f"Show #1, based on your favorites, is '{title1}' and it is about {description1}. "
        f"Show #2, based on my recommendations, is '{title2}' and it is about {description2}.\n"
        f"Here are also the 2 TV show ads. Hope you like them!")

    save_and_open_image(image_show_create_url, "show_create.png")
    save_and_open_image(recommended_show_image_url, "recommended_show.png")

if __name__ == "__main__":
    main()
