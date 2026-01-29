import torch
from .model import SimpleCNN
from .preprocess import transform_image, get_character

_worker_model = None

#Singleton pattern/Lazy loader for the model.Ensures the model is initialized only once per worker process.
def get_worker_model():
    global _worker_model
    if _worker_model is None:
        print("[Worker] Loading model (CPU)...")
        device = torch.device("cpu")
        model = SimpleCNN(num_classes=49)
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
        _worker_model = model
    return _worker_model

#Task entry point for the RQ/Celery worker.Handles preprocessing, inference, and response formatting.
def predict_task(image_bytes: bytes):
    try:
        model = get_worker_model()
        
        tensor = transform_image(image_bytes)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        idx = predicted_idx.item()
        return {
            "prediction_id": idx,
            "character": get_character(idx),
            "confidence": round(confidence.item(), 4),
            "status": "success"
        }
    except Exception as e:
        print(f"[Worker Error] {e}")
        return {"status": "error", "message": str(e)}
