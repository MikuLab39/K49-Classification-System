# 1. Base Image 
FROM python:3.9-slim 
 
WORKDIR /app 
 
# 2. Install System Deps (Minimal) 
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/* 
 
# 3. Install Python Deps (CPU Only) 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
 
# 4. Copy Code & Assets 
COPY ./src ./src 
COPY model.pth . 
COPY ./static ./static 
 
# 5. Env Vars 
ENV PYTHONPATH=/app 
ENV CUDA_VISIBLE_DEVICES="" 
 
# 6. Default Command (API) 
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
