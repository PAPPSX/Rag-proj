import streamlit as st
import torch
from llm_chains import chat, chat_pdf, handle_image
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from audio_handler import transcribe_audio
from pdf_handler import add_to_db
from html_templates import get_bot_template, get_user_template, css
import yaml
import os
from pydub import AudioSegment
import io

# Load configuration from yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the chain (to process the chat)
def load_chain(chat_history):
    # Check if PDF chat mode is selected
    if st.session_state.pdf_chat:
        st.write("Loading PDF chat chain...")
        return chat_pdf(chat_history)

    # Assuming 'user_id', 'input', and 'transcribe' are available from the session or UI
    user_id = "some_user_id"  # Retrieve the user_id appropriately
    input_text = st.session_state.user_question if st.session_state.user_question != "" else ""  # Retrieve input text
    transcribe_text = st.session_state.history[-1]['content'] if len(st.session_state.history) > 0 else ""  # If you want to pass transcribe text from chat history

    return chat(chat_id=st.session_state.session_key, user_id=user_id, input=input_text, transcribe=transcribe_text)

# Clear the input field after sending the message
def clear_input_field():
    if st.session_state.user_question == "":
        st.session_state.user_question = st.session_state.user_input
        st.session_state.user_input = ""

# Set the flag to send input
def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

# Toggle PDF chat mode
def toggle_pdf_chat():
    st.session_state.pdf_chat = True

# Save chat history
def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)

# Main function to run the app
def main():
    st.title("Multimodal Local Chat App")
    st.write(css, unsafe_allow_html=True)
    
    # Sidebar for selecting chat sessions
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # Selecting the current session
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)

    # Load the chat history
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")

    # Text input field
    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)

    voice_recording_column, send_button_column = st.columns(2)
    chat_container = st.container()
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", just_once=True)
    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, key="pdf_upload", type=["pdf"], on_change=toggle_pdf_chat)

    # Handle PDF upload
    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            add_documents_to_db(uploaded_pdf)

    # Handle audio upload
    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        st.write(f"Transcribed Audio: {transcribed_audio}")  # Show transcribed audio on the UI
        llm_response = load_chain(chat_history)  # Get response from LLM
        st.write("LLM Response: ", llm_response)  # Display LLM response on the UI

    # Handle voice recording
    if voice_recording and "bytes" in voice_recording:
        with st.spinner("Processing audio recording..."):
            if voice_recording.get("format") == "webm":
                webm_audio = AudioSegment.from_file(io.BytesIO(voice_recording["bytes"]), format="webm")
                mp3_audio = io.BytesIO()
                webm_audio.export(mp3_audio, format="mp3")
                transcribed_audio = transcribe_audio(mp3_audio.getvalue())
            else:
                transcribed_audio = transcribe_audio(voice_recording["bytes"])

            chat_history.add_user_message(f"Audio Recording: {transcribed_audio}")
            llm_response = load_chain(chat_history)
            chat_history.add_ai_message(llm_response)
            st.write(f"LLM Response: {llm_response}")  # Display response on the UI

    # Handle image upload
    if uploaded_image:
        with st.spinner("Processing image..."):
            user_message = "Describe this image in detail please."
            if st.session_state.user_question != "":
                user_message = st.session_state.user_question
                st.session_state.user_question = ""
            llm_answer = handle_image(uploaded_image.getvalue(), user_message)
            chat_history.add_user_message(user_message)
            chat_history.add_ai_message(llm_answer)
            st.write(f"LLM Response for Image: {llm_answer}")  # Show image response

    # Handle text input and generate response
    if st.session_state.user_question != "":
        llm_response = load_chain(chat_history)
        st.session_state.user_question = ""
        st.write(f"LLM Response: {llm_response}")  # Show response for text input

    st.session_state.send_input = False

    # Display chat history
    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in reversed(chat_history.messages):
                if message.type == "human":
                    st.write(get_user_template(message.content), unsafe_allow_html=True)
                else:
                    st.write(get_bot_template(message.content), unsafe_allow_html=True)

    save_chat_history()

if __name__ == "__main__":
    main()
