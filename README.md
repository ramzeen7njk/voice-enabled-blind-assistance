🎙️ Voice-Enabled Blind Assistance - Python Code
This Python script is the backbone of the Voice-Enabled Blind Assistance project, combining face detection, recognition, and real-time voice feedback to assist visually impaired users.

🧩 Features
Face Detection: Uses Haar Cascade and face recognition for identifying faces in real-time.
Voice Feedback: Announces known individuals and alerts for unknown faces via text-to-speech.
Face Enrollment: Allows adding new faces with voice commands.
Mobile Camera Integration: Captures video feed from an external mobile device.
Data Persistence: Stores face encodings and names in an Excel file for reuse.
🛠️ Key Libraries
cv2 for image processing
face_recognition for face encoding and matching
pyttsx3 for text-to-speech conversion
speech_recognition for voice commands
pandas for data storage
🚀 How to Run
Install the required libraries:
pip install opencv-python face_recognition pyttsx3 SpeechRecognition pandas
Replace the IP address in the main() function with your mobile device's IP for video feed.
Run the script:
python face.py  
Use voice commands like "Save as [name]" to register new faces or press q to quit.
📂 Files Generated
known_faces.xlsx: Stores details of registered faces.
face_encodings/: Saves face encoding files.
known_faces/: Saves face images.
🌟 Why It Matters
This script demonstrates the use of Python and IoT for assistive technologies, ensuring safety and convenience for visually impaired individuals.

