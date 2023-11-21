from bs4 import BeautifulSoup
import http.client
import json
import numpy as np
import requests
import ssl
import wave
from moviepy.editor import VideoFileClip
import whisper
from pydub import AudioSegment
import time
import torch
import streamlit as st

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    st.title("Legislative Transcription App")

    url = st.text_input("Enter the URL of the video to transcribe", 
                        "https://malegislature.gov/Events/Sessions/Detail/4512")
    
    if st.button("Transcribe"):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the video source tag
            video_source_tag = soup.find('source', type="video/mp4")

            if video_source_tag:
                video_url = video_source_tag['src']

                # Parse the video URL to get the hostname and path
                video_url_parts = video_url.split("/")
                hostname = video_url_parts[2]
                path = "/" + "/".join(video_url_parts[3:])

                # Introduce a delay before making the request (e.g., 2 seconds)
                time.sleep(2)

                # Use http.client to make the request
                conn = http.client.HTTPSConnection(hostname)
                conn.request("GET", path)
                video_response = conn.getresponse()

                # Check the status of the response
                if video_response.status == 200:
                    # Download the video
                    try:
                        with open("video.mp4", "wb") as f:
                            f.write(video_response.read())  
                    except Exception as e:
                        st.write(f"Error: {str(e)}")
                else: 
                        st.write(f"Error: {video_response.status} {video_response.reason}")
        

                # Extract audio from the video using moviepy
                clip = VideoFileClip("video.mp4")
                audio = clip.audio

                # Convert to pydub AudioSegment
                audio = AudioSegment.from_file("video.mp4", format="mp4")

                #Load the pre-trained model
                whisper_model = whisper.load_model("tiny.en")

                #convert audio to float type
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)

                #Transcribe the audio using whisper
                transcript = whisper_model.transcribe(audio_data)

                #Extract transcribed text from segments
                segments = transcript.get("segments", [])
                for segment in segments:
                    text = segment.get("text", "")
                    if text:
                        st.write(f"Transcribed text: {text}")

                st.write("Process completed successfully.")
            else:
                st.write("Error: Video source tag not found.")

if __name__ == "__main__":
    main()



