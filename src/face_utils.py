import face_recognition
import numpy as np
from PIL import Image

def detect_faces(image: Image.Image) -> list:
    """检测图片中的所有人脸位置"""
    img_np = np.array(image.convert("RGB"))
    face_locations = face_recognition.face_locations(img_np)
    return face_locations

def encode_face(image: Image.Image, face_location: tuple = None) -> np.ndarray:
    """对单张人脸进行128维特征编码"""
    img_np = np.array(image.convert("RGB"))
    if face_location:
        face_encodings = face_recognition.face_encodings(img_np, [face_location])
    else:
        face_encodings = face_recognition.face_encodings(img_np)
    return face_encodings[0] if face_encodings else None

def load_known_faces(known_faces_dir: str = "known_faces") -> tuple:
    """加载已知人脸库，返回编码和姓名列表"""
    known_encodings = []
    known_names = []
    import os
    for filename in os.listdir(known_faces_dir):
        if filename.endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            img = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
            encoding = face_recognition.face_encodings(img)[0]
            known_encodings.append(encoding)
            known_names.append(name)
    return known_encodings, known_names

def recognize_face(unknown_encoding: np.ndarray, known_encodings: list, known_names: list) -> str:
    """比对人脸编码，返回识别结果"""
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding)
    name = "Unknown"
    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]
    return name
