import cv2
import numpy as np
from PIL import Image

# 加载 OpenCV 自带的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image: Image.Image) -> list:
    """用 OpenCV 检测人脸，返回 (x,y,w,h) 格式的框"""
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # 转换成 face_recognition 格式 (top, right, bottom, left)
    face_locations = [(y, x+w, y+h, x) for (x,y,w,h) in faces]
    return face_locations

# 暂时注释掉识别相关函数，避免导入 dlib
# def encode_face(...):
# def load_known_faces(...):
# def recognize_face(...):    name = "Unknown"
    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]
    return name
