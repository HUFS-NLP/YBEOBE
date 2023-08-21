from datasets import load_dataset
import requests
import json
import uuid
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load the dataset
data = load_dataset("go_emotions", "simplified")

# Count total characters in the dataset
total_chars = sum(len(text) for text in data['train']['text'])

# Set a threshold
CHAR_THRESHOLD = 1_500_000

# if total_chars > CHAR_THRESHOLD:
#     raise ValueError(f"Total characters in the dataset is {total_chars:,}.")

# Azure Translator API Configuration
load_dotenv()
SUBSCRIPTION_KEY = os.environ.get("AZURE_TRANSLATOR_KEY")
if not SUBSCRIPTION_KEY:
    raise ValueError("No API key found.")
ENDPOINT = "https://api.cognitive.microsofttranslator.com"

def translate_go_emotions_to_korean(text):
    # API endpoint for translating text
    path = '/translate?api-version=3.0'
    params = '&to=ko'
    constructed_url = ENDPOINT + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
        'Ocp-Apim-Subscription-Region': 'eastasia',
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{'text': text}]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    print(response)

    # Assuming only one translation is returned and taking the first one
    translated_text = response[0]['translations'][0]['text']
    return translated_text

# Rate limiting
RATE_LIMIT_PAUSE = 1

# Take a subset of the data for testing
subset_data = data['train']['text'][:10]

# Translate all texts in the dataset
translated_texts = []
# for text in data['train']['text']:
for text in subset_data:
    translated_texts.append(translate_go_emotions_to_korean(text))
    time.sleep(RATE_LIMIT_PAUSE)

# Create a DataFrame and save to csv
df = pd.DataFrame({
    'text': translated_texts, 
    'labels': data['train']['labels'][:len(translated_texts)], 
    'id': data['train']['id'][:len(translated_texts)]
    })

df.to_csv('go_emotions_translated.csv', index=False)
