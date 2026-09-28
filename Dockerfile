FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend MP_WEB_DIR=/app/frontend
WORKDIR /app
COPY requirements.txt requirements-ai.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Optional Transformer backend: docker build --build-arg WITH_AI=1 .  (CPU-only PyTorch, much smaller than the CUDA build)
ARG WITH_AI=0
RUN if [ "$WITH_AI" = "1" ]; then       pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu &&       pip install --no-cache-dir -r requirements-ai.txt ; fi
COPY backend ./backend
COPY frontend ./frontend
COPY tools ./tools
COPY settings ./settings
COPY database ./database
COPY render_start.py ./render_start.py
RUN useradd -r -u 10001 mp && mkdir -p /data && chown -R mp /data
USER mp
CMD ["python", "render_start.py"]
