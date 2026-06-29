FROM python:3.11-slim
WORKDIR /project
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY ml ./ml
COPY models ./models
COPY backend ./backend
WORKDIR /project/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
