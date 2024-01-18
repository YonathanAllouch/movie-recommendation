from openai import OpenAI , RateLimitError
import openai
import os
import time
from PIL import Image
import requests
from io import BytesIO


# Load the API key from the environment variable
try: 
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"An error occurred: {e}")
    raise   

def generate_show_description(basis, shows):
    prompt = f"Create a name and a description (that is allowed by the safety system of openai) for a new TV show based on the following {basis} shows: {', '.join(shows)}. Present you reponse in this format 'the name you choose : the description you write'"
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "system", "content": "You are a TV show writer."},
    {"role": "user", "content": prompt}])
    return response.choices[0].message.content


def describe_picture_for_story(story_description, retries=2, delay=20):
    
    # Create a prompt for the GPT model
    prompt = f"Write a description of a picture that effectively illustrates the following story: {story_description}"

    for attempt in range(retries):
        try:
            # Call the OpenAI API
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",  # or another engine of your choice
                prompt=prompt,
                max_tokens=100  # You can adjust the number of tokens as needed
            )

            # Extract and return the picture description
            picture_description = response.choices[0].text.strip()
            return picture_description
        except openai.RateLimitError as e:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise e  # Reraise the exception if all retries fail
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


def generate_show_image(description, retries=3, delay=60):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    for attempt in range(retries):
        try:
            response = client.images.generate(
                prompt=description,
                size="1024x1024",   
                quality = "hd",
                n=1,
            )
            return response.data[0].url
        except openai.RateLimitError as e :
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise e  
        except openai.OpenAIError as e:
            raise e  
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

'''
story = describe_picture_for_story("a dog eat a cat")   
print(story)
image_url = generate_show_image(story)
print(image_url)
save_and_open_image(image_url, "test.jpg")
'''