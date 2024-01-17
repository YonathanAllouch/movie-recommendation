import openai
import os
from PIL import Image
import requests
from io import BytesIO


# Load the API key from the environment variable
try: 
    openai.api_key = os.getenv('OPENAI_API_KEY')
except Exception as e:
    print(f"An error occurred: {e}")
    raise   

def generate_show_description(basis, shows):
    prompt = f"Create a name and a description (that is allowed by the safety system of openai) for a new TV show based on the following {basis} shows: {', '.join(shows)}. Present you reponse in this format 'the name you choose : the description you write'"
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()


def generate_show_image(description):
   
    try:
        prompt = f"Create an image for a TV show described as: {description} , if the policy don't allow you to create the image just create an image that follow the policy ."
        response = openai.Image.create(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            )
        image_url = response.data[0].url
        return image_url
    except openai.error.InvalidRequestError as e:
        print(f"An error occurred: {e}")
        return None


def save_and_open_image(image_url, filename):
    try:
        # Fetch and save the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image.save(filename)

        # Open the image file
        os.system(f"open {filename}")  
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")