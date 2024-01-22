import pytest
import openai
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv
from unittest.mock import patch, Mock , MagicMock
from ShowSuggesterAI import load_tv_shows, get_embeddings, save_embeddings, load_embeddings
from closest_match_show import match_shows
from show_recommendation import calculate_average_vector, recommend_shows
from generate_TV_show import generate_show_description, generate_show_image, save_and_open_image

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

def test_get_embeddings():
    # Convert the mock data to the format expected by get_embeddings
    mock_tv_shows_data = {
        'Show 1': 'Description 1',
        'Show 2': 'Description 2',
        'Show 3': 'Description 3'
    }

    # Create a mock OpenAI client
    class MockOpenAIClient:
        def __init__(self):
            self.embeddings = MockEmbeddings()

    class MockEmbeddings:
        def create(self, input, model):
            # Ensure the mock response is consistent with the number of inputs
            return MockResponse(len(input))

    class MockResponse:
        def __init__(self, num_items):
            self.data = [MockEmbedding() for _ in range(num_items)]

    class MockEmbedding:
        def __init__(self):
            self.embedding = [0.1, 0.2, 0.3]

    mock_openai_client = MockOpenAIClient()
    # Mock the OpenAI client
    with patch('openai.OpenAI', return_value=mock_openai_client):
        # Call the get_embeddings function
        embeddings = get_embeddings(mock_tv_shows_data, mock_openai_client)

        # Assert that the embeddings dictionary is correctly generated
        expected_embeddings_dict = {
            'Show 1': [0.1, 0.2, 0.3],
            'Show 2': [0.1, 0.2, 0.3],
            'Show 3': [0.1, 0.2, 0.3]
        }
        assert embeddings == expected_embeddings_dict

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

### Test generate_TV_show.py

def test_generate_show_description():
    # Mock response content
    mock_response_content = "Mock Show Name: This is a mock description."

    class MockMessage:
        def __init__(self):
            self.content = mock_response_content

    class MockCompletion:
        def __init__(self):
            self.message = MockMessage()

    class MockResponse:
        def __init__(self):
            self.choices = [MockCompletion()]

    with patch('openai.ChatCompletion.create', return_value=MockResponse()):
        description = generate_show_description("favorite", ["Show 1", "Show 2"])

        # Check if the returned description contains a colon, indicating the format "name : description"
        assert ":" in description


def test_generate_show_image():
    # Mock URL for testing
    mock_url = "https://example.com/mock-image.jpg"

    class MockImageData:
        def __init__(self):
            self.url = mock_url

    class MockImageResponse:
        def __init__(self):
            self.data = [MockImageData()]

    with patch('openai.Image.create', return_value=MockImageResponse()):
        actual_url = generate_show_image("Mock Title", "Mock Description")

        # Check if the returned value is a string
        assert isinstance(actual_url, str)

        # Check if the returned string is a valid URL format
        assert actual_url.startswith("http://") or actual_url.startswith("https://")


def test_save_and_open_image():
    mock_image_url = "https://example.com/image.jpg"
    mock_filename = "test_image.jpg"

    # Read an actual small image file for the mock response content
    with open("/Users/yonathanallouch/Desktop/HW2 software/EX2-Embedding/small_image_test.jpg", "rb") as img_file:  # Use a real image file path
        mock_image_content = img_file.read()

    with patch('requests.get', MagicMock(return_value=MagicMock(content=mock_image_content))) as mock_get, \
         patch('os.system') as mock_os_system:
        save_and_open_image(mock_image_url, mock_filename)

        mock_get.assert_called_with(mock_image_url)
        mock_os_system.assert_called_with(f"open {mock_filename}")
