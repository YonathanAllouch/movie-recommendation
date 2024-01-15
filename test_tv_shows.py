import pytest
import os
from dotenv import load_dotenv
from unittest.mock import patch
from ShowSuggesterAI import load_tv_shows, get_embeddings, save_embeddings, load_embeddings

# Simple test to ensure pytest is running correctly
def test_example():
    assert 1 == 1

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
