import pytest
import numpy as np
import os
from dotenv import load_dotenv
from unittest.mock import patch
from ShowSuggesterAI import load_tv_shows, get_embeddings, save_embeddings, load_embeddings
from closest_match_show import match_shows
from show_recommendation import calculate_average_vector, recommend_shows


### Test ShowSuggesterAI.py
    
# Fixture for the CSV file path
@pytest.fixture
def tv_shows_csv():
    return '/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/imdb_tvshows.csv'  

# Test loading TV shows from a CSV file
def test_load_tv_shows(tv_shows_csv):
    shows = load_tv_shows(tv_shows_csv)
    assert isinstance(shows, dict)
    assert len(shows) > 0  # Ensuring the dictionary is not empty

# Test getting embeddings (API call is mocked)
# Example of improved test for get_embeddings with error handling
@patch('ShowSuggesterAI.openai')
def test_get_embeddings_with_error_handling(mock_openai):
    mock_openai.Embedding.create.side_effect = Exception("API error")
    api_key = "test_api_key"
    with pytest.raises(Exception) as excinfo:
        get_embeddings({'Test Show': 'A test description.'}, api_key)
    assert "API error" in str(excinfo.value)

# Test saving embeddings to a file using pickle
def test_save_embeddings(tmpdir):
    embeddings = {'Test Show': [0.1, 0.2, 0.3]}
    filepath = tmpdir.join("embeddings.pkl")
    save_embeddings(embeddings, filepath)
    assert os.path.exists(filepath)

# Test loading embeddings from a pickle file
def test_load_embeddings(tmpdir):
    embeddings = {'Test Show': [0.1, 0.2, 0.3]}
    filepath = tmpdir.join("embeddings.pkl")
    save_embeddings(embeddings, filepath)
    loaded_embeddings = load_embeddings(filepath)
    assert embeddings == loaded_embeddings

# Test that no API call is made if embeddings are already saved
@patch('ShowSuggesterAI.get_embeddings')
def test_no_api_call_if_embeddings_saved(mock_get_embeddings, tv_shows_csv, tmpdir):
    # Ensure that embeddings are saved
    shows = load_tv_shows(tv_shows_csv)
    save_embeddings({'Test Show': [0.1, 0.2, 0.3]}, tmpdir / "embeddings.pkl")

    # Load embeddings and check that get_embeddings was not called
    load_embeddings(tmpdir / "embeddings.pkl")
    mock_get_embeddings.assert_not_called()


### Test closest_match_show.py
def test_match_shows():
    known_shows = ["Game of Thrones", "Lupin", "The Witcher"]
    input_shows = ["gem of throns", "lupan", "witcher"]
    expected_output = ["Game of Thrones", "Lupin", "The Witcher"]
    assert match_shows(input_shows, known_shows) == expected_output

def test_match_shows_empty_input():
    known_shows = ["Game of Thrones", "Lupin", "The Witcher"]
    input_shows = []
    expected_output = []
    assert match_shows(input_shows, known_shows) == expected_output

def test_match_shows_no_match():
    known_shows = ["Game of Thrones", "Lupin", "The Witcher"]
    input_shows = ["Random Show", "Another Show"]
    # The function now correctly returns the input string if no match is found
    assert match_shows(input_shows, known_shows, threshold=60) == input_shows

def test_match_shows_special_characters():
    known_shows = ["Game of Thrones", "Lupin", "The Witcher"]
    input_shows = ["Game@of Thrones", "Lupin!"]
    expected_output = ["Game of Thrones", "Lupin"]
    assert match_shows(input_shows, known_shows) == expected_output

def test_match_shows_case_insensitivity():
    known_shows = ["Game of Thrones", "Lupin", "The Witcher"]
    input_shows = ["game of thrones", "LUPIN"]
    expected_output = ["Game of Thrones", "Lupin"]
    assert match_shows(input_shows, known_shows) == expected_output

def test_match_shows_partial_matches():
    known_shows = ["Game of Thrones", "Lupin", "The Witcher"]
    input_shows = ["Game", "Witch"]
    # Assuming the function returns the closest match for partial strings
    expected_output = ["Game of Thrones", "The Witcher"]
    assert match_shows(input_shows, known_shows) == expected_output

   
### Test show_recommendation.py    

def test_calculate_average_vector():
    embeddings = {
        "Show1": np.array([1, 2, 3]),
        "Show2": np.array([4, 5, 6]),
        "Show3": np.array([7, 8, 9])
    }
    shows = ["Show1", "Show2"]
    expected_average = np.array([2.5, 3.5, 4.5])
    np.testing.assert_array_equal(calculate_average_vector(shows, embeddings), expected_average)

def test_recommend_shows():
    embeddings = {
        "Show1": np.array([1, 2, 3]),
        "Show2": np.array([4, 5, 6]),
        "Show3": np.array([7, 8, 9]),
        "Show4": np.array([1, 3, 5]),
        "Show5": np.array([6, 7, 8]),
        "Show6": np.array([2, 4, 6])
    }
    user_shows = ["Show1", "Show2"]
    all_shows = list(embeddings.keys())
    recommendations = recommend_shows(user_shows, all_shows, embeddings)
    
    # Check if the recommendations are at most 5 and user's shows are not in recommendations
    assert len(recommendations) <= 5
    assert all(show not in user_shows for show in recommendations.keys())


