from io import BytesIO
import os
from typing import Optional
import requests
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import chromadb
from pdf_handler import create_embeddings, load_vectordb, add_to_db
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

app = FastAPI()

class ChatRequest(BaseModel):
    chat_id: int
    user_id: int
    input: str
    transcribe: Optional[str] = None

class ChatPdfRequest(BaseModel):
    chat_id: int
    user_id: int
    input: str
    doc_path: Optional[str] = None

class HandleImageRequest(BaseModel):
    user_message: str

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    history = []

    if request.transcribe:
        if request.transcribe.strip():  
            history.append(
                {
                    "role": "system",
                    "content": f"You are a friendly assistant. Generate responses based on the user's input and the context of the conversation. Additionally, process the provided transcription text: '{request.transcribe}'",
                }
            )
        else:
            history.append(
                {
                    "role": "system",
                    "content": "You are a friendly assistant. Generate responses based on the user's input and the context of the conversation. The user has not provided any transcription text.",
                }
            )
    else:
        history.append(
            {
                "role": "system",
                "content": "You are a friendly assistant. Generate responses based on the user's input and the context of the conversation. No transcription text has been provided.",
            }
        )


    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": "You are a friendly assistant"},
                {"role": "user", "content": request.input},
            ]
        },
    )

    response_content = response.json()['result']['response']
    history.append({"role": "assistant", "content": response_content})
    return JSONResponse(content={"response": response_content})

@app.post("/chat_pdf")
async def chat_pdf(request: ChatPdfRequest):
    client = chromadb.HttpClient(host='localhost', port=9000)
    collection = client.get_or_create_collection(name="test")
    history = []

    if request.doc_path:
        add_to_db(request.doc_path, request.chat_id, collection)

    vector_data = collection.query(query_texts=[request.input], n_results=5, where={"chat_id": request.chat_id})["documents"]

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
                {"role": "user", "content": request.input},
            ]
        },
    )

    response_content = response.json()['result']['response']
    history.append({"role": "assistant", "content": response_content})
    return JSONResponse(content={"response": response_content})

@app.post("/handle_image")
async def handle_image(image_path: UploadFile = File(...), user_message: str = ""):
    try:
        # Read the uploaded file
        img_file = await image_path.read()
        img = Image.open(BytesIO(img_file))
        
        # Convert the image to an integer array
        img = image_to_int_array(img)
        
        # Make the request to Cloudflare API
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
                "image": img
            },
        )

        response.raise_for_status()  # Raise an error for bad responses
        response_content = response.json().get('result', {}).get('response', 'No response from the model.')
        
        return JSONResponse(content={"response": response_content})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/transcribe_audio")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()
    transcribe = transcribe_audio(audio_bytes)
    return JSONResponse(content={"transcription": transcribe})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5173)
# To run the app, use this command
# uvicorn <filename>:app --reload
