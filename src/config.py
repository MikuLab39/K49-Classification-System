import os

# Redis 
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_NAME = "default"

# dataset statistics (EDA)
STATS_MEAN = (0.1801,)
STATS_STD = (0.3421,)
