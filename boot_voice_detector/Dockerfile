FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Копирование проекта
COPY . .

# Создание директорий
RUN mkdir -p data/raw data/processed models

# Запуск приложения
CMD ["python3", "scripts/train_cnn_spectrogram.py"]