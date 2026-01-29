import sys
from redis import Redis
from rq import Worker, Queue
from .config import REDIS_URL, QUEUE_NAME
from .tasks import get_worker_model # Pre-loading
from rq import SimpleWorker #No fork

def start_worker():
    print(f"[Worker] Connecting to Redis at {REDIS_URL}...")
    try:
        conn = Redis.from_url(REDIS_URL)
        queue = Queue(QUEUE_NAME, connection=conn)
        
        print("[Worker] Pre-loading model into RAM (Parent Process)...")
        get_worker_model() 
        print("[Worker] Model loaded. Ready to fork.")

        worker = SimpleWorker([queue], connection=conn)
        
        worker.work(max_jobs=50)
        
    except Exception as e:
        print(f"[Worker] Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    start_worker()