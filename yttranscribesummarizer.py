import streamlit as st
from dotenv import load_dotenv

load_dotenv() ##load all the nevironment variables
import os
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import whisper
import torch
from pytube import YouTube

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt="""You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  """

destination = "."  ## '.' means it store in current folder
final_filename = "extract_audio"
root_dir = os.getcwd()

## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript
    
    except TranscriptsDisabled:
        #Set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the model
        whisper_model = whisper.load_model("large", device=device)
        # Get the video
        video = YouTube(youtube_video_url)

        # Convert video to Audio
        audio = video.streams.filter(only_audio=True).first()

        # Save to destination
        output = audio.download(output_path = destination)

        _, ext = os.path.splitext(output)
        new_file = final_filename + '.mp3'

        # Change the name of the file
        os.rename(output, new_file)
        print(youtube_video_url, youtube_link)

        # Run the test
        audio_file = root_dir+new_file
        result = whisper_model.transcribe(audio_file)

        anylanguage_to_english = whisper_model.transcribe(audio_file, task = 'translate')

        # Show the result
        print(anylanguage_to_english["text"])
        
        return anylanguage_to_english["text"]

    
    except Exception as e:
        raise e


## getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text, safety_settings={'HARM_CATEGORY_HATE_SPEECH':'block_none','HARM_CATEGORY_HARASSMENT':'block_none'})
    print(response.prompt_feedback)
    return response.text
    #return response.parts

st.title("YouTube Transcript to Summary Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Notes"):
    transcript_text=extract_transcript_details(youtube_link)
    # Open the file in read mode
    # Open the file in read mode with explicit encoding
    #with open('C:\\Users\\Praveen\\geminiapp\\transcribed_text.txt', 'r', encoding='utf-8') as file:
    # Read the contents of the file and store it in a variable
    #    transcript_text = file.read()

    #print(transcript_text)
    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("## Detailed Notes:")
        st.write(summary)



