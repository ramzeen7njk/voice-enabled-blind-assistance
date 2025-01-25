import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
import pyttsx3
import speech_recognition as sr
import threading
import time
import pickle
import queue
import requests

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognition
r = sr.Recognizer()

# Create a queue for speech tasks
speech_queue = queue.Queue()

# Create folders to store images and encodings if they don't exist
if not os.path.exists('known_faces'):
    os.makedirs('known_faces')
if not os.path.exists('face_encodings'):
    os.makedirs('face_encodings')

# Create or load the Excel file
excel_file = 'known_faces.xlsx'
try:
    df = pd.read_excel(excel_file)
    if 'Encoding_File' not in df.columns:
        df = pd.DataFrame(columns=['Name', 'Timestamp', 'Encoding_File'])
        df.to_excel(excel_file, index=False)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    df = pd.DataFrame(columns=['Name', 'Timestamp', 'Encoding_File'])
    df.to_excel(excel_file, index=False)

def clean_dataframe(df):
    # Remove rows with non-string Encoding_File values
    df = df[df['Encoding_File'].apply(lambda x: isinstance(x, str))]
    # Remove rows where the encoding file doesn't exist
    df = df[df['Encoding_File'].apply(os.path.exists)]
    return df

df = clean_dataframe(df)

# Load known face encodings
known_face_encodings = []
known_face_names = []

for _, row in df.iterrows():
    encoding_file = row['Encoding_File']
    if isinstance(encoding_file, str) and os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as f:
            encoding = pickle.load(f)
        known_face_encodings.append(encoding)
        known_face_names.append(row['Name'])
    else:
        print(f"Warning: Encoding file '{encoding_file}' not found or invalid for {row['Name']}")

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def speak_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

def speak(text):
    speech_queue.put(text)

# Start the speech worker thread
speech_thread = threading.Thread(target=speak_worker, daemon=True)
speech_thread.start()

def listen_for_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        speak("Listening for command...")
        audio = r.listen(source)
        try:
            command = r.recognize_google(audio).lower()
            print(f"Command recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
            speak("Could not understand audio. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            speak("There was an error processing your request. Please try again.")
            return None

def save_face(frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected in the image using Haar Cascade.")
        speak("No face detected in the image. Please try again.")
        return False
    
    (x, y, w, h) = faces[0]
    face_image = frame[y:y+h, x:x+w]
    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    face_encodings = face_recognition.face_encodings(rgb_face)
    
    if not face_encodings:
        print("Could not encode the face.")
        speak("Could not encode the face. Please try again.")
        return False
    
    face_encoding = face_encodings[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"known_faces/{name}_{timestamp}.jpg"
    cv2.imwrite(image_filename, face_image)
    
    # Save face encoding
    encoding_filename = f"face_encodings/{name}_{timestamp}.pkl"
    with open(encoding_filename, 'wb') as f:
        pickle.dump(face_encoding, f)
    
    # Save to DataFrame and Excel
    df.loc[len(df)] = [name, timestamp, encoding_filename]
    df.to_excel(excel_file, index=False)
    
    # Update known faces
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    
    print(f"Face saved as {name}")
    speak(f"Face saved as {name}")
    return True

def process_frame(frame):
    global process_this_frame, face_locations, face_encodings, face_names
    
    if process_this_frame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            face_names.append(name)
    
    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown faces
        else:
            color = (0, 255, 0)  # Green for known faces
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    return frame

def voice_assistant():
    global face_names
    last_announcement = ""
    while True:
        if face_names:
            current_faces = set(face_names)
            if "Unknown" in current_faces:
                current_faces.remove("Unknown")
            
            if current_faces:
                announcement = f"Known person{'s' if len(current_faces) > 1 else ''} approaching: {', '.join(current_faces)}"
                if announcement != last_announcement:
                    speak(announcement)
                    last_announcement = announcement
            elif last_announcement:
                speak("No known persons in view")
                last_announcement = ""
        
        time.sleep(5)  # Check every 5 seconds

# Start voice assistant thread
voice_thread = threading.Thread(target=voice_assistant, daemon=True)
voice_thread.start()

def get_frame(url, max_retries=5, retry_delay=2, timeout=10):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)
            return frame
        except requests.RequestException as e:
            print(f"Error connecting to mobile camera (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Please check your connection and try again.")
                return None

def main():
    # Replace with your mobile device's IP address and port
    ip_address = "192.168.1.6"  # Example IP, replace with your actual IP
    port = "8080"  # Default port for IP Webcam app
    url = f"http://{ip_address}:{port}/shot.jpg"

    print(f"Attempting to connect to mobile camera at {url}")
    print("Press 'q' to quit, 's' to save a new face")

    while True:
        frame = get_frame(url)
        if frame is not None:
            frame = process_frame(frame)
            cv2.imshow("Assistive Face Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                speak("Please say 'save as' followed by the person's name")
                command = listen_for_command()
                if command and command.startswith("save as"):
                    name = command.split("save as")[-1].strip()
                    if save_face(frame, name):
                        print(f"Face saved successfully as {name}")
                    else:
                        print("Failed to save face. Please try again.")
                        speak("Failed to save face. Please try again.")
        else:
            print("Failed to get frame. Retrying...")
            time.sleep(1)

    cv2.destroyAllWindows()

    # Stop the speech worker thread
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()
