from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from typing import List
from redis import Redis
from rq import Queue
from rq.job import Job
import torch
import time

from src.model import SimpleCNN
from src.preprocess import transform_image, get_character
from src.config import REDIS_URL, QUEUE_NAME
from src.tasks import predict_task 

# 打开 src/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
# [新增] 引入 StaticFiles 和 FileResponse
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import FileResponse

# Global State
api_model = None
redis_conn = None
task_queue = None
DEVICE = torch.device("cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle."""
    global api_model, redis_conn, task_queue
    
    # 1. Initialize Inference Engine
    try:
        print(f"[API] Loading sync model...")
        api_model = SimpleCNN(num_classes=49).to(DEVICE)
        api_model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
        api_model.eval()
    except Exception as e:
        print(f"[API] Warning: Local model failed to load ({e}). /predict will fail.")

    # 2. Initialize Task Queue
    try:
        redis_conn = Redis.from_url(REDIS_URL)
        task_queue = Queue(QUEUE_NAME, connection=redis_conn)
        print(f"[API] Connected to Redis.")
    except Exception as e:
        print(f"[API] Error connecting to Redis: {e}")

    yield
    print("[API] Shutting down...")

app = FastAPI(title="K49 API (Sync/Async)", version="2.0", lifespan=lifespan)

# [新增] 1. 挂载 static 目录，用于提供 css/js/html
app.mount("/static", StaticFiles(directory="static"), name="static")

# [新增] 2. 覆盖根路由，返回我们刚写的 HTML 页面
@app.get("/")
async def read_index():
    # 这会读取 static/index.html 并返回给浏览器
    return FileResponse("static/index.html")

@app.get("/")
def health_check():
    return {"status": "ok", "mode": "production"}

# Synchronous Endpoints
@app.post("/predict")
async def predict_sync(file: UploadFile = File(...)):
    if api_model is None:
        raise HTTPException(503, "Model not loaded")
    
    start_time = time.time()
    image_bytes = await file.read()
    
    tensor = transform_image(image_bytes).to(DEVICE)
    
    with torch.no_grad():
        outputs = api_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    idx = predicted_idx.item()
    return {
        "prediction_id": idx,
        "character": get_character(idx),
        "confidence": round(confidence.item(), 4),
        "time_ms": round((time.time() - start_time) * 1000, 2)
    }

# Asynchronous Endpoints
@app.post("/batch_predict")
async def predict_async(files: List[UploadFile] = File(...)):
    if task_queue is None:
        raise HTTPException(503, "Redis not connected")
        
    task_ids = []
    for file in files:
        content = await file.read()
        job = task_queue.enqueue(predict_task, content)
        task_ids.append(job.get_id())
    
    return {"message": "Queued", "task_ids": task_ids}

@app.get("/tasks/{task_id}")
def get_task_result(task_id: str):
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except:
        raise HTTPException(404, "Task not found")
        
    if job.is_finished:
        return {"status": "completed", "result": job.result}
    elif job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)}
    else:
        return {"status": "processing"}
