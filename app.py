import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

# 加载 OpenCV 自带的人脸检测器（无需 dlib，无需额外安装）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image: Image.Image) -> list:
    """
    核心功能：用 OpenCV 检测人脸，返回 face_recognition 格式的坐标 (top, right, bottom, left)
    输入：PIL Image 对象
    输出：人脸位置列表
    """
    # 转换为 OpenCV 支持的格式
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 检测人脸（scaleFactor=1.1 是检测精度，minNeighbors=5 是去噪）
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # 转换坐标格式：OpenCV (x,y,w,h) → face_recognition (top, right, bottom, left)
    face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
    return face_locations

# ===================== Streamlit 界面配置 =====================
st.set_page_config(page_title="人脸检测应用", page_icon="🧑‍🤝‍🧑", layout="wide")
st.title("🧑‍🤝‍🧑 人脸检测 Web 应用")
st.sidebar.header("操作面板")

# 侧边栏功能：上传图片 / 使用示例图片
uploaded_file = st.sidebar.file_uploader("上传图片（支持jpg/png/jpeg）", type=["jpg", "png", "jpeg"])
use_sample = st.sidebar.checkbox("使用示例图片", value=False)

# 处理图片加载逻辑
if use_sample:
    # 示例图片：奥巴马头像（无需本地文件，直接联网加载）
    image = Image.open("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Barack_Obama.jpg/440px-Barack_Obama.jpg")
elif uploaded_file:
    # 加载用户上传的图片
    image = Image.open(uploaded_file)
else:
    # 初始提示
    st.info("👉 请在左侧面板上传图片，或勾选「使用示例图片」开始检测")
    st.stop()

# ===================== 人脸检测 + 结果展示 =====================
# 展示原始图片
st.subheader("📸 原始图片")
st.image(image, use_column_width=True)

# 执行人脸检测
face_locations = detect_faces(image)
st.subheader(f"✅ 检测结果：共识别到 {len(face_locations)} 张人脸")

# 绘制人脸框并展示
img_with_boxes = image.copy()
draw = ImageDraw.Draw(img_with_boxes)
for (top, right, bottom, left) in face_locations:
    # 画红色人脸框（宽度3像素）
    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)

# 展示带人脸框的图片
st.image(img_with_boxes, use_column_width=True)

# 可选：展示人脸坐标（调试用）
if st.sidebar.checkbox("显示人脸坐标", value=False):
    st.write("人脸位置坐标（top, right, bottom, left）：")
    st.write(face_locations)        if not known_encodings:
            st.warning("请在 known_faces 目录下添加已知人脸图片（命名为姓名.jpg）")
        else:
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_img = image.crop((left, top, right, bottom))
                encoding = encode_face(face_img)
                if encoding is not None:
                    name = recognize_face(encoding, known_encodings, known_names)
                    st.write(f"人脸 {i+1}: {name}")
    except Exception as e:
        st.error(f"识别失败: {e}")
