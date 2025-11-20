import uvicorn
import numpy as np
import base64
import cv2
import os
import io
import nest_asyncio
import gdown 
from pyngrok import ngrok
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from deepface import DeepFace

# --- PYTORCH IMPORTS ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# --- LINE SDK ---
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction
from linebot.exceptions import InvalidSignatureError
from linebot.models.events import PostbackEvent

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
LINE_CHANNEL_ACCESS_TOKEN = 'yMTfcTZoEaG2kSZMDtUCVT5I8S47c0APKUNUtRvFfIVfAj+005EixdA9iDJPDReJaM8snIwnjMgPJ3B/rYru1Fr6/veFTAhga+DXB/97zSfoMo279kisRv1hsKM6K+0Me32GvqQvG07qCPMXuHda9QdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'b8c65e65a4ead4ef817d7c66f2832e0c'
LINE_HOST_USER_ID = 'U669226ca0e16195477ca5857a469567d'

GDRIVE_FILE_ID = "1RtR1gTpcWGPY3z05hhwtdr4zxlMuxkzP"
SPOOF_MODEL_PATH = "resnet50_spoof_best.pt"
NGROK_AUTH_TOKEN = '35HRFySeHKxSBkuFZ48n0tT6sZl_CoVKLmVZ1o1CuKUwoSje' # ใส่ Token

# ==========================================
# 🧠 MODEL (SpoofNet)
# ==========================================
class SpoofNet(nn.Module):
    def __init__(self):
        super(SpoofNet, self).__init__()
        self.pretrained_net = resnet50(weights=None) 
        self.features = nn.Sequential(
            self.pretrained_net.conv1, self.pretrained_net.bn1, self.pretrained_net.relu, self.pretrained_net.maxpool,
            self.pretrained_net.layer1, self.pretrained_net.layer2, self.pretrained_net.layer3, self.pretrained_net.layer4
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

# Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spoof_model = None
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_pytorch_model():
    global spoof_model
    if not os.path.exists(SPOOF_MODEL_PATH):
        try:
            gdown.download(f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}', SPOOF_MODEL_PATH, quiet=False)
        except: pass
    
    try:
        model = SpoofNet()
        if os.path.exists(SPOOF_MODEL_PATH):
            checkpoint = torch.load(SPOOF_MODEL_PATH, map_location=device)
            sd = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            model.load_state_dict(sd)
            model.to(device).eval()
            spoof_model = model
            print("✅ Model Loaded!")
    except Exception as e: print(f"❌ Error loading model: {e}")

load_pytorch_model()

# ==========================================
# 🚀 APP SETUP & DB
# ==========================================
try:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)
except: line_bot_api = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

DB_PATH = "database"
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)

# 🔥🔥 GLOBAL DB 🔥🔥
user_status_db = {} 
known_face_db = {}

# --- Helpers ---
def base64_to_pil_image(s):
    if "," in s: s = s.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert('RGB')

def base64_to_cv2_image(s):
    if "," in s: s = s.split(",")[1]
    arr = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def load_known_faces():
    emb = {}
    if not os.path.exists(DB_PATH): return emb
    for f in os.listdir(DB_PATH):
        if f.endswith(".npy"):
            name = f.split("_")[0]
            try:
                e = np.load(os.path.join(DB_PATH, f))
                if name not in emb: emb[name] = []
                for i in e: emb[name].append(i)
            except: pass
    return emb
known_face_db = load_known_faces()

def calculate_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class RequestModel(BaseModel):
    image_data: str = None; name: str = None; user_id: str = None; images: List[str] = None

# ==========================================
# 🔌 API ENDPOINTS
# ==========================================

@app.post("/api/v1/spoof-check")
async def spoof_check(req: RequestModel):
    if spoof_model is None: return {"is_real": True, "confidence": 1.0, "mode": "mock"}
    try:
        img = base64_to_pil_image(req.image_data)
        t = preprocess_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = spoof_model(t)
            score = out.item()
            is_real = score > 0.5 
            conf = score if is_real else (1 - score)
        return {"is_real": is_real, "confidence": conf}
    except: return {"is_real": True, "confidence": 1.0}

@app.post("/api/v1/check-face-existence")
async def check_face_existence(req: RequestModel):
    try:
        img = base64_to_cv2_image(req.image_data)
        objs = DeepFace.represent(img, model_name="VGG-Face", detector_backend="opencv", enforce_detection=False)
        if not objs: return {"found": False}
        target = objs[0]["embedding"]
        found = None
        for n, embs in known_face_db.items():
            for e in embs:
                if calculate_cosine_similarity(target, e) > 0.60:
                    found = n; break
            if found: break
        return {"found": True, "name": found} if found else {"found": False}
    except: return {"found": False}

@app.post("/api/v1/request-permission")
async def request_permission(req: RequestModel):
    user_id = f"user_{np.random.randint(10000, 99999)}"
    user_status_db[user_id] = "pending"
    print(f"👉 NEW REQ: {user_id} (pending)")

    if line_bot_api:
        try:
            actions = [
                PostbackAction(label="อนุมัติ", data=f"action=approve&user_id={user_id}"),
                PostbackAction(label="ปฏิเสธ", data=f"action=reject&user_id={user_id}")
            ]
            template = TemplateSendMessage(
                alt_text="คำขอใหม่",
                template=ButtonsTemplate(title=f"คำขอ: {req.name}", text=f"ID: {user_id}", actions=actions)
            )
            line_bot_api.push_message(LINE_HOST_USER_ID, template)
        except Exception as e: print(f"LINE Error: {e}")
    
    return {"success": True, "user_id": user_id}

@app.get("/api/v1/check-approval-status/{user_id}")
async def check_approval_status(user_id: str):
    status = user_status_db.get(user_id, "pending")
    # print(f"🔍 Checking {user_id}: {status}") # Uncomment to debug spam
    return {"status": status}

@app.post("/api/v1/register-faces")
async def register_faces(req: RequestModel):
    embs = []
    for s in req.images:
        try:
            img = base64_to_cv2_image(s)
            objs = DeepFace.represent(img, model_name="VGG-Face", detector_backend="opencv", enforce_detection=False)
            if objs: embs.append(objs[0]["embedding"])
        except: pass
    if embs:
        np.save(os.path.join(DB_PATH, f"{req.name}_{req.user_id}_embeddings.npy"), np.array(embs))
        global known_face_db; known_face_db = load_known_faces()
        return {"success": True}
    return {"success": False}

@app.post("/api/v1/scan-face")
async def scan_face(req: RequestModel):
    try:
        img = base64_to_cv2_image(req.image_data)
        objs = DeepFace.represent(img, model_name="VGG-Face", detector_backend="opencv", enforce_detection=False)
        if not objs: return {"is_match": False}
        target = objs[0]["embedding"]
        best_score = 0; best_name = None
        for n, embs in known_face_db.items():
            for e in embs:
                s = calculate_cosine_similarity(target, e)
                if s > best_score: best_score = s; best_name = n
        if best_score > 0.60: return {"is_match": True, "user": {"name": best_name}, "confidence": float(best_score)}
        else: return {"is_match": False}
    except: return {"is_match": False}

# --- 🔥 WEBHOOK HANDLER 🔥 ---
@app.post("/webhook")
async def line_webhook(request: Request):
    if not webhook_handler: return {"status": "mock"}
    sig = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try: webhook_handler.handle(body.decode(), sig)
    except InvalidSignatureError: raise HTTPException(status_code=401)
    return {"status": "ok"}

@webhook_handler.add(PostbackEvent)
def handle_postback(event):
    # แกะ data เช่น "action=approve&user_id=user_12345"
    raw = event.postback.data
    print(f"📩 WEBHOOK DATA: {raw}")
    
    data = dict(x.split("=") for x in raw.split("&"))
    uid = data.get("user_id")
    act = data.get("action")

    if uid and act:
        # อัปเดต DB
        user_status_db[uid] = "approved" if act == "approve" else "rejected"
        print(f"✅ UPDATE DB: {uid} -> {user_status_db[uid]}")
        
        # ตอบกลับ
        msg = f"อนุมัติ {uid} เรียบร้อย" if act == "approve" else f"ปฏิเสธ {uid} แล้ว"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

# Run
nest_asyncio.apply()
if NGROK_AUTH_TOKEN == 'YOUR_NGROK_AUTH_TOKEN_HERE':
    print("⚠️ ใส่ Token ก่อน!")
else:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    ngrok.kill()
    tunnel = ngrok.connect(8000)
    print("Public URL:", tunnel.public_url)
    uvicorn.run(app, port=8000)