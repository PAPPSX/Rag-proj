import os
import requests
import yaml
import chromadb
from dotenv import load_dotenv
from pdf_handler import create_embeddings, load_vectordb, add_to_db
from PIL import Image
from image_handler import image_to_int_array
from audio_handler import transcribe_audio

load_dotenv()

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
API_ID = os.getenv("CLOUDFLARE_AI_API")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

models = config["llm_model"]
image_model = config["Image_model"]

def chat(chat_id, user_id, input, transcribe):
    history = []

    if transcribe:
        if transcribe.strip():  
            history.append(
                {
                    "role": "system",
                    "content": f"You are a friendly assistant, generate responses based on the user's input and the context of the conversation. content: {transcribe}",
                }
            )
        else:
            history.append(
                {
                    "role": "system",
                    "content": "You are a friendly assistant, generate responses based on the user's input and the context of the conversation.",
                }
            )
    else:
        history.append(
            {
                "role": "system",
                "content": "You are a friendly assistant, generate responses based on the user's input and the context of the conversation.",
            }
        )

    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": "You are a friendly assistant"},
                {"role": "user", "content": input},
            ]
        },
    )

    print(response.json()['result']['response'])
    history.append({"role": "assistant", "content": response.json()['result']['response']})
    return response.json()['result']['response']

def chat_pdf(chat_id, user_id, input, doc_path: str | None = None):
    client = chromadb.HttpClient(host='localhost', port=9000)
    collection = client.get_or_create_collection(name="test")
    history = []

    if doc_path:
        add_to_db(doc_path, chat_id, collection)

    vector_data = collection.query(query_texts=[input], n_results=5, where={"chat_id": chat_id})["documents"]

    history.append(
        {
            "role": "system",
            "content": f"You are a friendly assistant, generate responses based on the user's input, document data, and the context of the conversation. Context: {vector_data}",
        }
    )
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": history + [
                {"role": "user", "content": input},
            ]
        },
    )
    print(response.json()['result']['response'])
    history.append({"role": "assistant", "content": response.json()['result']['response']})
    return response.json()['result']['response']

def handle_image(image_path, user_message):
    img = Image.open(image_path)
    img = image_to_int_array(img)
    # API request to Cloudflare AI (Image description)
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{image_model}",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly assistant that describes images accurately and in detail. You generate responses based on the user's input and the context of the conversation.",
                },
                {"role": "user", "content": user_message},
            ],
            "image":img
        },
    )

    print(response.json()['result']['response'])
    return response.json()['result']['response']


# chat(1, 1, "Hello", None)

# chat_pdf(1, 1, "Tell me about Mokshad Sankhe")

# image_path = "./rag.jpg"
# handle_image(image_path, "Describe this image in detail please.")

# with open("harvard.wav", "rb") as f:
#     audio_bytes = f.read()
# transcribe = transcribe_audio(audio_bytes)
# print(transcribe)
# chat(1, 1, "Explain what is mentioned in audio in detail in easy words: " + transcribe, transcribe)