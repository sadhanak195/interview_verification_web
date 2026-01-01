import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# ---------------- Paths ----------------
FACE_CASCADE = 'haar_face.xml'
TEMPLATE_SHIRT_DIR = 'templates/shirts'
TEMPLATE_TIE_DIR = 'templates/ties'
CAPTURED_FRAMES_DIR = 'captured_frames'
os.makedirs(CAPTURED_FRAMES_DIR, exist_ok=True)

# ---------------- Load Haar and Templates ----------------
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)

def load_templates(folder, size=(50,50)):
    templates = []
    for f in os.listdir(folder):
        temp = cv2.imread(os.path.join(folder,f),0)
        if temp is not None:
            temp = cv2.resize(temp,size)
            templates.append(temp)
    return templates

shirt_templates = load_templates(TEMPLATE_SHIRT_DIR)
tie_templates = load_templates(TEMPLATE_TIE_DIR)

def match_templates(region_gray, templates, threshold=0.75):
    max_conf = 0
    detected = False
    for temp in templates:
        if region_gray.shape[0]<temp.shape[0] or region_gray.shape[1]<temp.shape[1]:
            continue
        res = cv2.matchTemplate(region_gray,temp,cv2.TM_CCOEFF_NORMED)
        conf = np.max(res)
        if conf >= threshold:
            detected=True
        if conf>max_conf:
            max_conf = conf
    return detected, max_conf

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Interview Verification", layout="wide")
st.title("💼 Interview Verification Dashboard")

# Sidebar Controls
run_webcam = st.sidebar.button("Start Webcam")
stop_webcam = st.sidebar.button("Stop Webcam")

if 'run_flag' not in st.session_state:
    st.session_state['run_flag'] = False
if run_webcam:
    st.session_state['run_flag'] = True
if stop_webcam:
    st.session_state['run_flag'] = False

# Webcam Capture
cap = cv2.VideoCapture(0)
frame_count = 0
FRAME_PLACEHOLDER = st.empty()

# Gallery
st.subheader("Captured Frames")
gallery = st.columns(6)

while st.session_state['run_flag']:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    shirt_detected, tie_detected, final_conf = False, False, 0

    for (x,y,w,h) in faces:
        # Crop upper body below face
        y1 = y+h
        y2 = min(y+h+int(h*1.5), frame.shape[0])
        shirt_region = frame[y1:y2,x:x+w]
        shirt_gray = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2GRAY)

        shirt_detected, shirt_conf = match_templates(shirt_gray, shirt_templates)
        tie_detected, tie_conf = match_templates(shirt_gray, tie_templates)

        final_conf = 100 if (shirt_detected and tie_detected) else 0

        # Draw rectangles (subtle colors)
        if shirt_detected:
            cv2.rectangle(frame,(x,y1),(x+w,y2),(0,128,255),2)  # orange-ish
        if tie_detected:
            tie_x1=x+w//3
            tie_x2=x+2*w//3
            cv2.rectangle(frame,(tie_x1,y1),(tie_x2,y2),(255,0,0),2)  # blue

        # Save frames if both detected
        if final_conf==100:
            path = f'{CAPTURED_FRAMES_DIR}/frame_{frame_count}.jpg'
            cv2.imwrite(path,frame)
            frame_count+=1

    # Status overlay with subtle gray background
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(640,40),(50,50,50),-1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay,alpha,frame,1-alpha,0)
    text = f"Shirt: {shirt_detected} | Tie: {tie_detected} | Confidence: {final_conf}%"
    cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    # Display frame in Streamlit
    FRAME_PLACEHOLDER.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    # Display captured frames gallery
    paths = sorted(os.listdir(CAPTURED_FRAMES_DIR))
    for idx, p in enumerate(paths[-6:]):
        img = Image.open(os.path.join(CAPTURED_FRAMES_DIR,p))
        gallery[idx].image(img, use_column_width=True)
