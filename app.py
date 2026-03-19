import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

# 加载 OpenCV 自带的人脸检测器（无额外依赖）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image: Image.Image) -> list:
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [(y, x+w, y+h, x) for (x, y, w, h) in faces]

# ===================== Streamlit 界面 =====================
st.title("🧑‍🤝‍🧑 人脸检测 Web 应用")
st.sidebar.header("操作面板")

uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
use_sample = st.sidebar.checkbox("使用示例图片", value=False)

if use_sample:
    image = Image.open("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Barack_Obama.jpg/440px-Barack_Obama.jpg")
elif uploaded_file:
    image = Image.open(uploaded_file)
else:
    st.info("请上传图片或勾选示例图片开始检测")
    st.stop()

# 展示原图
st.subheader("📸 原始图片")
st.image(image, use_column_width=True)

# 检测并画框
face_locations = detect_faces(image)
st.subheader(f"✅ 检测到 {len(face_locations)} 张人脸")

img_with_boxes = image.copy()
draw = ImageDraw.Draw(img_with_boxes)
for (top, right, bottom, left) in face_locations:
    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)

st.image(img_with_boxes, use_column_width=True)
