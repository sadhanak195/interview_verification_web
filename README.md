# Interview Verification Web App

Professional Streamlit app for verifying if a person is wearing a formal shirt and tie during an online interview.

## Features
- Live webcam detection
- Shirt & tie detection with confidence
- Captured frames gallery
- Professional dashboard interface

## How to Run

1. **Extract Templates**  
   - Download or clone the repository.  
   - Extract `templates.zip` in the same folder as `app.py`.  
     After extraction, the structure should be:
     ```
     interview_verification_streamlit/
     │
     ├── app.py
     ├── haar_face.xml
     ├── templates/
     │   ├── shirts/
     │   └── ties/
     └── captured_frames/  (will be created automatically)

2. **Install Dependencies**  

3. **Run the App**  

4. **Open in Browser**  
- The dashboard will open automatically.  
- Use the sidebar buttons to **Start** and **Stop** webcam.  
- Frames are only saved when **both shirt and tie are detected**.  
- Captured frames are saved in the `captured_frames/` folder.  
