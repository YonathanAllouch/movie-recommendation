from openai import OpenAI 
import openai
import os
import time
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_show_description(basis, shows):
    prompt = f"Create a name and a description for a new TV show based on the following {basis} shows: {', '.join(shows)}. Present you reponse in this format 'the name you choose : the description you write'"
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
    {"role": "system", "content": "You are a TV show writer."},
    {"role": "user", "content": prompt}])
    return response.choices[0].message.content



def generate_show_image(title , description, retries=3, delay=60):
    prompt = (f"Act as an expert in TV show ads creator. "
              f"Please help me to create an amazing ad for the new TV show I created that will illustrate the following details - CREATE THE ACTUAL IMAGE AD: "
              f"Title: {title}. Description: {description}. Please include the title name in the image.")


    for attempt in range(retries):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",   
                quality = "standard",
                n=1
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

