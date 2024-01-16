from fuzzywuzzy import process
import pandas as pd

# Load the CSV file containing TV show titles
file_path = '/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/imdb_tvshows.csv'
tv_shows_df = pd.read_csv(file_path)

# Extract the list of known shows from the CSV file
known_shows = tv_shows_df['Title'].tolist()

from fuzzywuzzy import process
import pandas as pd

def match_shows(input_shows, known_shows, threshold=60):
    """ Matches input shows with known shows using fuzzy matching. """
    matched_shows = []
    for show in input_shows:
        match, score = process.extractOne(show, known_shows)
        if score >= threshold:
            matched_shows.append(match)
        else:
            matched_shows.append(show)  # or use 'None' or any other placeholder
    return matched_shows

def get_favorite_tv_shows(known_shows):
    while True:
        input_shows = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: ")
        shows = [show.strip() for show in input_shows.split(',')]

        if len(shows) > 1:
            matched_shows = match_shows(shows, known_shows)
            confirmation = input(f"Just to make sure, do you mean {', '.join(matched_shows)}? (y/n) ")

            if confirmation.lower() == 'y':
                print("Great! Generating recommendations...")
                return matched_shows
            else:
                print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.")
        else:
            print("Please enter more than one TV show.")

# Example usage
'''favorite_shows = get_favorite_tv_shows()'''
