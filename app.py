import streamlit as st
from PIL import Image, ImageDraw
from src.face_utils import detect_faces

st.title("🧑‍🤝‍🧑 人脸检测 Web 应用")
st.sidebar.header("操作面板")

uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
use_sample = st.sidebar.checkbox("使用示例图片", value=False)

if use_sample:
    image = Image.open("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Barack_Obama.jpg/440px-Barack_Obama.jpg")
elif uploaded_file:
    image = Image.open(uploaded_file)
else:
    st.info("请上传图片或选择示例图片开始检测")
    st.stop()

st.subheader("原始图片")
st.image(image, use_column_width=True)

st.subheader("人脸检测结果")
face_locations = detect_faces(image)
st.write(f"检测到 {len(face_locations)} 张人脸")

# 画人脸框
img_with_boxes = image.copy()
draw = ImageDraw.Draw(img_with_boxes)
for (top, right, bottom, left) in face_locations:
    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
st.image(img_with_boxes, use_column_width=True)
# （可选）人脸识别
if st.sidebar.checkbox("识别人脸（需加载人脸库）"):
    st.subheader("人脸识别结果")
    try:
        known_encodings, known_names = load_known_faces()
        if not known_encodings:
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
