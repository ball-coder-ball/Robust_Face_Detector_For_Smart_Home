import uvicorn
import numpy as np
import base64
import cv2
import os
import io
import nest_asyncio
import gdown 
import uuid
from pyngrok import ngrok
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from deepface import DeepFace

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, URIAction
from linebot.exceptions import InvalidSignatureError
from linebot.models.events import PostbackEvent

LINE_CHANNEL_ACCESS_TOKEN = 'uN65Uh43/s9MDQXmPxQHaCsU2I/Fz7JA9cm+qd1Gq2CDLcRpoV7FNMAW1My1XMNEaM8snIwnjMgPJ3B/rYru1Fr6/veFTAhga+DXB/97zSezswAP1KKCoUFVpMt83dQVLEoBKdMa7hA/FhRoi3wowwdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'b8c65e65a4ead4ef817d7c66f2832e0c'
LINE_HOST_USER_ID = 'U669226ca0e16195477ca5857a469567d'

GDRIVE_FILE_ID = "1RtR1gTpcWGPY3z05hhwtdr4zxlMuxkzP"
SPOOF_MODEL_PATH = "resnet50_spoof_best.pt"
NGROK_AUTH_TOKEN = '35HRFySeHKxSBkuFZ48n0tT6sZl_CoVKLmVZ1o1CuKUwoSje' 

class SpoofNet(nn.Module):
    def __init__(self):
        super(SpoofNet, self).__init__()
        self.pretrained_net = resnet50(weights=None) 
        self.features = nn.Sequential(
            self.pretrained_net.conv1,
            self.pretrained_net.bn1,
            self.pretrained_net.relu,
            self.pretrained_net.maxpool,
            self.pretrained_net.layer1,
            self.pretrained_net.layer2,
            self.pretrained_net.layer3,
            self.pretrained_net.layer4
        )
        self.conv2d = nn.Conv2d(2048, 32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x) 
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spoof_model = None

preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def download_model_if_needed():
    if not os.path.exists(SPOOF_MODEL_PATH):
        print(f"⬇️ Downloading model...")
        try:
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, SPOOF_MODEL_PATH, quiet=False)
            print("✅ Downloaded!")
        except Exception as e:
            print(f"❌ Download failed: {e}")

def load_pytorch_model():
    global spoof_model
    download_model_if_needed()
    
    print(f"🔄 Loading SpoofNet...")
    try:
        model = SpoofNet()
        if os.path.exists(SPOOF_MODEL_PATH):
            checkpoint = torch.load(SPOOF_MODEL_PATH, map_location=device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            spoof_model = model
            print("✅ Model Loaded!")
        else:
            print(f"❌ Model file not found")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

load_pytorch_model()

try:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)
except:
    line_bot_api = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_PATH = "database"
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
PUBLIC_URL = ""

user_status_db = {} 
known_face_db = {}

def base64_to_pil_image(base64_string):
    if "," in base64_string: base64_string = base64_string.split(",")[1]
    img_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

    body = await request.body()
    try: webhook_handler.handle(body.decode(), signature)
    except InvalidSignatureError: raise HTTPException(status_code=401)
    return {"status": "ok"}

@webhook_handler.add(PostbackEvent)
def handle_postback(event):
    data = dict(x.split("=") for x in event.postback.data.split("&"))
    uid = data.get("user_id")
    act = data.get("action")
    
    print(f"📩 [WEBHOOK] Received: ID={uid}, Action={act}")

    if uid and act:
        user_status_db[uid] = "approved" if act == "approve" else "rejected"
        print(f"✅ [DB UPDATE] {uid} is now {user_status_db[uid]}")
        
        msg_text = f"อนุมัติ {uid} เรียบร้อยแล้ว" if act == "approve" else f"ปฏิเสธคำขอ {uid} แล้ว"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg_text))

nest_asyncio.apply()
if NGROK_AUTH_TOKEN == 'YOUR_NGROK_AUTH_TOKEN_HERE':
    print("\n⚠️ กรุณาใส่ Ngrok Token ก่อนรันครับ!\n")
else:
    try:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        ngrok.kill()
        tunnel = ngrok.connect(8000)
        PUBLIC_URL = tunnel.public_url
        print("Public URL:", PUBLIC_URL)
        uvicorn.run(app, port=8000)
    except Exception as e:
        print(f"Ngrok Error: {e}")