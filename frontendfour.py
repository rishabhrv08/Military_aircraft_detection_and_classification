import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Military Aircraft Classifier", layout="wide")

# Define the Adaptive SE Block and EfficientNetWithSE Model
class AdaptiveSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AdaptiveSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out).view(avg_out.size(0), -1, 1, 1)
        spatial_out = self.spatial_conv(x)
        spatial_out = self.sigmoid_spatial(spatial_out)
        out = x * avg_out
        out = out * spatial_out
        return out

class EfficientNetWithSE(nn.Module):
    def __init__(self, model):
        super(EfficientNetWithSE, self).__init__()
        self.features = model.features
        self.se_block = AdaptiveSEBlock(1280)
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.se_block(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model = EfficientNetWithSE(model)
    model.load_state_dict(torch.load(r"C:\\Users\\Acer\\Desktop\\Frontend\\efficientnet_model_with_adaptive_se.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ["A-10", "A-400M", "AG-600", "AH-64", "AV-8B", "An-124", "An-22", "An-225", "An-72", "B-1", 
               "B-2", "B-21", "B-52", "Be-200", "C-130", "C-17", "C-2", "C-390", "C-5", "CH-47", "CL-415", "E-2",
               "E-7", "EF-2000", "F-117", "F-14", "F-15", "F-16", "F-22", "F-35", "F-4", "F/A-18", "H-6", "J-10", 
               "J-20", "J-35", "JAS-39", "JF-17", "JH-7", "KAAN", "KC-135", "KF-21", "KJ-600", "Ka-27", "Ka-52", 
               "MQ-9", "Mi-24", "Mi-26", "Mi-28", "Mi-8", "Mig-29", "Mig-31", "Mirage2000", "P-3", "RQ-4", "Rafale", 
               "SR-71", "Su-24", "Su-25", "Su-34", "Su-57", "TB-001", "TB-2", "Tornado", "Tu-160", "Tu-22M", "Tu-95", 
               "U-2", "UH-60", "US-2", "V-22", "Vulcan", "WZ-7", "XB-70", "Y-20", "YF-23", "Z-19"]

st.markdown(
    """
    <style>
    .stApp {
        background-color: #c1dfff;
    }
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .title {
        font-size: 20em;
        color: #007bff;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 10em !important;
    }
    .stFileUploader>div {
        border: 2px dashed #007bff;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>Military Aircraft Classification</h1>", unsafe_allow_html=True)
st.write("Choose an option below to classify military aircraft from images or videos.")

option = st.selectbox("Select Classification Type", ["Image Classification", "Video Classification"])

if option == "Image Classification":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = class_names[predicted_idx.item()]
        st.success(f"Predicted Class: {predicted_class}")

elif option == "Video Classification":
    @st.cache_resource
    def load_yolo():
        return torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s')

    yolo_model = load_yolo()
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_video.read())
            temp_file_path = temp_file.name

        video = cv2.VideoCapture(temp_file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = "processed_video.mp4"
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(video.get(3)), int(video.get(4))))

        progress_bar = st.progress(0)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            results = yolo_model(frame)
            labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            for label, coord in zip(labels, coordinates):
                if coord[4] > 0.5:
                    x1, y1, x2, y2 = int(coord[0] * frame.shape[1]), int(coord[1] * frame.shape[0]), int(coord[2] * frame.shape[1]), int(coord[3] * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cropped_image = frame[y1:y2, x1:x2]
                    if cropped_image.size != 0:
                        pil_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                        image_tensor = transform(pil_cropped_image).unsqueeze(0)
                        with torch.no_grad():
                            output = model(image_tensor)
                            _, predicted_idx = torch.max(output, 1)
                            predicted_class = class_names[predicted_idx.item()]
                        cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)

            current_frame += 1
            progress_bar.progress(min(current_frame / frame_count, 1.0))

        video.release()
        out.release()

        st.write("Processed video is ready for download:")
        with open(out_path, 'rb') as f:
            st.download_button("Download Processed Video", data=f, file_name="processed_video.mp4", mime="video/mp4")
        os.remove(temp_file_path)
        os.remove(out_path)
