# hw03 人脸识别 Web 应用

## 项目结构
hw03/
├── src/ # 核心代码
│ └── face_utils.py # 人脸检测 / 编码 / 识别工具
├── app.py # Streamlit Web 应用
├── requirements.txt # 依赖清单
├── known_faces/ # 已知人脸库（可选）
└── README.md # 本文档

## 功能说明
1.  **人脸检测**：使用 `face_recognition` 库检测图片中的所有人脸，并绘制红色框标注位置。
2.  **人脸识别（可选）**：加载 `known_faces` 目录下的已知人脸图片，对检测到的人脸进行比对识别。
3.  **Web 界面**：基于 Streamlit 实现，支持上传图片或使用示例图片，直观展示检测/识别结果。

## 环境准备
1.  安装依赖：
    ```bash
    pip install -r requirements.txt
