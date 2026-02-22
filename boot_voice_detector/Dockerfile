FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask gunicorn

COPY . .

RUN mkdir -p saved_models results static

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"]