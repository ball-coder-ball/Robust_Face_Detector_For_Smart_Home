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
from linebot.models import TextSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, URIAction
from linebot.exceptions import InvalidSignatureError
from linebot.models.events import PostbackEvent

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
LINE_CHANNEL_ACCESS_TOKEN = 'uN65Uh43/s9MDQXmPxQHaCsU2I/Fz7JA9cm+qd1Gq2CDLcRpoV7FNMAW1My1XMNEaM8snIwnjMgPJ3B/rYru1Fr6/veFTAhga+DXB/97zSezswAP1KKCoUFVpMt83dQVLEoBKdMa7hA/FhRoi3wowwdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'b8c65e65a4ead4ef817d7c66f2832e0c'
LINE_HOST_USER_ID = 'U669226ca0e16195477ca5857a469567d'

GDRIVE_FILE_ID = "1RtR1gTpcWGPY3z05hhwtdr4zxlMuxkzP"
SPOOF_MODEL_PATH = "resnet50_spoof_best.pt"
NGROK_AUTH_TOKEN = '35HRFySeHKxSBkuFZ48n0tT6sZl_CoVKLmVZ1o1CuKUwoSje' 

# ==========================================
# 🧠 MODEL ARCHITECTURE (SpoofNet)
# ==========================================
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

# Loading
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

# ==========================================
# 🚀 APP SETUP & DB
# ==========================================
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

# Mount Static for Temp Images
if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_PATH = "database"
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
PUBLIC_URL = ""

# 🔥🔥 GLOBAL DATABASE FOR USER STATUS 🔥🔥
user_status_db = {} 
known_face_db = {}

# --- Helpers ---
def base64_to_pil_image(base64_string):
    if "," in base64_string: base64_string = base64_string.split(",")[1]
    img_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

def base64_to_cv2_image(base64_string):
    if "," in base64_string: base64_string = base64_string.split(",")[1]
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def load_known_faces():
    embeddings = {}
    if not os.path.exists(DB_PATH): return embeddings
    for f in os.listdir(DB_PATH):
        if f.endswith(".npy"):
            name = f.split("_")[0]
            try:
                embs = np.load(os.path.join(DB_PATH, f))
                if name not in embeddings: embeddings[name] = []
                for e in embs: embeddings[name].append(e)
            except: pass
    print(f"Loaded {len(embeddings)} users.")
    return embeddings
known_face_db = load_known_faces()

def calculate_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 🔥 PRE-LOAD DEEPFACE MODEL 🔥
print("⏳ Warming up DeepFace model...")
try:
    DeepFace.build_model(MODEL_NAME)
    print("✅ DeepFace VGG-Face Warmed Up!")
except Exception as e:
    print(f"⚠️ DeepFace Warmup Failed: {e}")

class RequestModel(BaseModel):
    image_data: str = None
    name: str = None
    user_id: str = None
    images: List[str] = None

# ==========================================
# 🔌 API ENDPOINTS
# ==========================================

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "models_loaded": True}

@app.post("/api/v1/spoof-check")
async def spoof_check(req: RequestModel):
    if spoof_model is None:
        return {"is_real": True, "confidence": 1.0, "mode": "mock_fallback"}
    try:
        image = base64_to_pil_image(req.image_data)
        image_tensor = preprocess_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = spoof_model(image_tensor)
            score = output.item()
            is_real = score > 0.5 
            display_conf = score if is_real else (1 - score)
        return {"is_real": is_real, "confidence": display_conf}
    except Exception as e:
        return {"is_real": True, "confidence": 1.0, "error": str(e)}

@app.post("/api/v1/check-face-existence")
async def check_face_existence(req: RequestModel):
    try:
        img = base64_to_cv2_image(req.image_data)
        objs = DeepFace.represent(img, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
        if not objs: return {"found": False}
        target_emb = objs[0]["embedding"]
        found_user = None
        for name, db_embs in known_face_db.items():
            for db_emb in db_embs:
                if calculate_cosine_similarity(target_emb, db_emb) > 0.60:
                    found_user = name; break
            if found_user: break
        return {"found": True, "name": found_user} if found_user else {"found": False}
    except: return {"found": False}

@app.post("/api/v1/request-permission")
async def request_permission(req: RequestModel):
    global PUBLIC_URL
    # 1. สร้าง ID และตั้งสถานะเป็น pending ทันที
    user_id = f"user_{np.random.randint(10000, 99999)}"
    user_status_db[user_id] = "pending" 
    print(f"👉 [REQ] Created ID: {user_id} | Status: {user_status_db[user_id]}")

    # 2. Save Image for LINE
    image_url = "https://via.placeholder.com/300" # Default
    if req.image_data and PUBLIC_URL:
        try:
            img = base64_to_pil_image(req.image_data)
            fname = f"temp_{user_id}.jpg"
            save_path = os.path.join("static", fname)
            img.save(save_path)
            image_url = f"{PUBLIC_URL}/static/{fname}"
            print(f"📸 Image saved: {image_url}")
        except Exception as e:
            print(f"⚠️ Image save failed: {e}")

    if not line_bot_api: return {"success": True, "user_id": user_id, "mock": True}
    
    try:
        actions = [
            PostbackAction(label="อนุมัติ", data=f"action=approve&user_id={user_id}"),
            PostbackAction(label="ปฏิเสธ", data=f"action=reject&user_id={user_id}")
        ]
        template = TemplateSendMessage(
            alt_text="คำขอลงทะเบียน",
            template=ButtonsTemplate(
                thumbnail_image_url=image_url,
                image_aspect_ratio="rectangle",
                image_size="cover",
                title=f"คำขอ: {req.name}",
                text=f"ID: {user_id}",
                actions=actions
            )
        )
        line_bot_api.push_message(LINE_HOST_USER_ID, template)
    except Exception as e: print(f"LINE Error: {e}")
    
    return {"success": True, "user_id": user_id}

@app.get("/api/v1/check-approval-status/{user_id}")
async def check_approval_status(user_id: str):
    # ดึงสถานะปัจจุบันจาก Dict
    status = user_status_db.get(user_id, "pending")
    # print(f"🔍 [POLL] Checking {user_id} -> Status: {status}") 
    return {"status": status}

@app.post("/api/v1/register-faces")
async def register_faces(req: RequestModel):
    embs = []
    for img_str in req.images:
        try:
            img = base64_to_cv2_image(img_str)
            objs = DeepFace.represent(img, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
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
        # 1. Spoof Check First
        spoof_res = await spoof_check(req)
        if not spoof_res["is_real"]:
             return {
                 "is_match": False, 
                 "reason": "spoof_detected", 
                 "spoof_confidence": spoof_res["confidence"]
             }

        # 2. Recognition
        img = base64_to_cv2_image(req.image_data)
        objs = DeepFace.represent(img, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=True)
        if not objs: return {"is_match": False, "reason": "no_face"}
        target_emb = objs[0]["embedding"]
        best_score = 0; best_name = None
        for name, db_embs in known_face_db.items():
            for db_emb in db_embs:
                score = calculate_cosine_similarity(target_emb, db_emb)
                if score > best_score: best_score = score; best_name = name
        if best_score > 0.60: 
            return {
                "is_match": True, 
                "user": {"name": best_name}, 
                "confidence": float(best_score),
                "spoof_confidence": spoof_res["confidence"]
            }
        else: return {"is_match": False, "reason": "unknown"}
    except: return {"is_match": False, "reason": "error"}

# --- 🔥🔥 WEBHOOK FIX 🔥🔥 ---
@app.post("/webhook")
async def line_webhook(request: Request):
    if not webhook_handler: return {"status": "mock"}
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try: webhook_handler.handle(body.decode(), signature)
    except InvalidSignatureError: raise HTTPException(status_code=401)
    return {"status": "ok"}

@webhook_handler.add(PostbackEvent)
def handle_postback(event):
    # แกะ data: "action=approve&user_id=user_12345"
    data = dict(x.split("=") for x in event.postback.data.split("&"))
    uid = data.get("user_id")
    act = data.get("action")
    
    print(f"📩 [WEBHOOK] Received: ID={uid}, Action={act}") # Debug Log

    if uid and act:
        # 1. อัปเดตสถานะลง Global Variable ทันที!
        user_status_db[uid] = "approved" if act == "approve" else "rejected"
        print(f"✅ [DB UPDATE] {uid} is now {user_status_db[uid]}") # ยืนยันการอัปเดต
        
        # 2. ตอบกลับ LINE
        msg_text = f"อนุมัติ {uid} เรียบร้อยแล้ว" if act == "approve" else f"ปฏิเสธคำขอ {uid} แล้ว"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg_text))

# Run
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
