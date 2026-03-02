"""
WHISPER - АВТОСОХРАНЕНИЕ ПРЯМО В ФАЙЛ
"""

import os
import torch
import whisper
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Оптимизации
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Проверка GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"⚠️  CPU MODE - БУДЕТ МЕДЛЕННО!")

# Загружаем модель один раз
print(f"\n📦 Загрузка модели medium...")
start_load = time.time()
model = whisper.load_model("medium", device=device)
print(f"   Загружено за {time.time()-start_load:.1f}с")

def transcribe_file(audio_path, language):
    """Быстрая транскрибация"""
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            fp16=True if device == "cuda" else False,
            verbose=False,
            temperature=0.0,
            without_timestamps=True,
            beam_size=1,
            best_of=1,
        )
        return result["text"].strip(), None
    except Exception as e:
        return "", str(e)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='ru')
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--save-interval', type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🔍 ПОИСК ФАЙЛОВ С ОШИБКОЙ")
    print(f"{'='*60}")

    # Пути
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "raw_annotations.csv")
    audio_root = os.path.join(project_root, "data")
    
    # Загрузка CSV
    print(f"\n📂 Загрузка {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "text" not in df.columns:
        df["text"] = ""
    
    # Сбор файлов с ошибкой
    files_to_process = []
    error_text = "[ERROR] Речь не распознана"
    
    for idx, row in df.iterrows():
        if args.max and len(files_to_process) >= args.max:
            break
        
        if str(row.get("text", "")) != error_text:
            continue
        
        rel_path = row.get("audio_path") or row.get("relative_path")
        if not isinstance(rel_path, str):
            continue
        
        audio_path = os.path.join(audio_root, rel_path.replace("\\", os.sep).replace("/", os.sep))
        
        if os.path.exists(audio_path):
            files_to_process.append((idx, audio_path))
    
    total = len(files_to_process)
    if total == 0:
        print(f"\n✅ Нет файлов с ошибкой")
        return
    
    print(f"\n📊 Найдено файлов: {total}")
    print(f"{'='*60}\n")
    
    # Прогрев
    if total > 0:
        print("🔥 Прогрев модели...")
        transcribe_file(files_to_process[0][1], args.language)
        print("   Готово\n")
    
    # Основной цикл
    start_time = time.time()
    processed = 0
    errors = 0
    
    for i, (idx, audio_path) in enumerate(files_to_process):
        # Прогресс
        elapsed = time.time() - start_time
        if processed > 0:
            speed = processed / elapsed
            eta = (total - processed) / speed
        else:
            speed = 0
            eta = 0
        
        print(f"\r📊 [{processed}/{total}] {speed:.1f} ф/с | ETA: {eta:.1f}с", end="")
        
        # Транскрибация
        text, error = transcribe_file(audio_path, args.language)
        
        if error:
            errors += 1
            print(f"\n❌ Ошибка: {error[:100]}")
        else:
            df.at[idx, "text"] = text
            processed += 1
        
        # АВТОСОХРАНЕНИЕ ПРЯМО В ФАЙЛ
        if (i + 1) % args.save_interval == 0:
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Автосохранено в {os.path.basename(csv_path)}")
    
    # Финальное сохранение
    print(f"\n\n💾 Финальное сохранение...")
    df.to_csv(csv_path, index=False)
    
    # Итог
    total_time = time.time() - start_time
    avg_speed = processed / total_time if total_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ ГОТОВО!")
    print(f"📊 Обработано: {processed}/{total}")
    print(f"❌ Ошибок: {errors}")
    print(f"⏱️  Время: {total_time:.1f}с")
    print(f"⚡ Скорость: {avg_speed:.1f} файлов/сек")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()