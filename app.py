# 第1行：导入streamlit
import streamlit as st
# 第2行：导入opencv
import cv2
# 第3行：导入numpy
import numpy as np
# 第4行：导入PIL（你确认过没错的这行）
from PIL import Image, ImageDraw

# 第5行：加载OpenCV人脸检测器（自带模型，无需下载）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 第6行：定义人脸检测函数
def detect_faces(image: Image.Image) -> list:
    # 第7行：转换为numpy数组
    img_np = np.array(image.convert("RGB"))
    # 第8行：转灰度图（人脸检测需要）
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # 第9行：检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # 第10行：转换坐标格式
    return [(y, x+w, y+h, x) for (x, y, w, h) in faces]

# 第11行：设置界面标题
st.title("🧑‍🤝‍🧑 人脸检测 Web 应用")
# 第12行：侧边栏标题
st.sidebar.header("操作面板")

# 第13行：上传图片控件
uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
# 第14行：示例图片勾选框
use_sample = st.sidebar.checkbox("使用示例图片", value=False)

# 第15行：图片加载逻辑
if use_sample:
    # 第16行：加载示例图片（联网）
    image = Image.open("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Barack_Obama.jpg/440px-Barack_Obama.jpg")
elif uploaded_file:
    # 第17行：加载上传的图片
    image = Image.open(uploaded_file)
else:
    # 第18行：初始提示
    st.info("请上传图片或勾选示例图片开始检测")
    # 第19行：停止执行
    st.stop()

# 第20行：展示原始图片标题
st.subheader("📸 原始图片")
# 第21行：展示原始图片
st.image(image, use_column_width=True)

# 第22行：执行人脸检测
face_locations = detect_faces(image)
# 第23行：展示检测结果标题
st.subheader(f"✅ 检测到 {len(face_locations)} 张人脸")

# 第24行：复制图片用于画框
img_with_boxes = image.copy()
# 第25行：创建绘图对象
draw = ImageDraw.Draw(img_with_boxes)
# 第26行：遍历人脸位置画框
for (top, right, bottom, left) in face_locations:
    # 第27行：画红色人脸框（宽度3像素）
    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)

# 第28行：展示带人脸框的图片
st.image(img_with_boxes, use_column_width=True)
