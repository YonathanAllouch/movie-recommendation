## ShowSuggesterAI

### Overview

ShowSuggesterAI is a Python-based application designed to offer personalized TV show recommendations based on user preferences. Leveraging advanced AI techniques and the OpenAI API, ShowSuggesterAI analyzes user-provided favorite shows to suggest similar content. Furthermore, it creatively generates descriptions and promotional images for entirely new, AI-conceived shows tailored to the user's tastes.

### Features

Personalized Recommendations: Generates TV show suggestions aligned with user preferences.
AI-Generated Show Concepts: Creates unique TV show ideas, complete with descriptions and promotional images.
Interactive User Experience: Engages users in selecting their favorite shows to tailor recommendations and generated content.

### Prerequisites

Before you begin, ensure you have the following installed:

Python 3.x
Required Python libraries: csv, re, pickle, os, dotenv, logging, and openai.

Additionally, you'll need:

An OpenAI API key set up as an environment variable (OPENAI_API_KEY).
The .env file configured with your OpenAI API key.

### Installation

Clone the repository:
```
git clone https://github.com/YonathanAllouch/movie-recommendation/master/ShowSuggesterAI.git
```
Navigate to the project directory:
```
cd ShowSuggesterAI
```
Install the required Python packages:
```
pip install -r requirements.txt
```

### Usage

To run ShowSuggesterAI, execute the following command in the terminal:
```
python ShowSuggesterAI.py
```
Follow the on-screen prompts to input your favorite TV shows and interact with the AI to receive personalized show suggestions and AI-generated show concepts.

### Contributing

Contributions to ShowSuggesterAI are welcome! Please read the CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests to us.

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.

