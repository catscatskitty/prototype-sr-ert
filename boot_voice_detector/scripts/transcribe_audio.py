"""
<<<<<<< HEAD
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
=======
WHISPER НА M4 - СЧЕТЧИК + АВТОСОХРАНЕНИЕ КАЖДЫЕ 50 ФАЙЛОВ
"""

import os
import sys
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

# Перехватываем весь вывод MLX
class NullWriter:
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass

original_stdout = sys.stdout
original_stderr = sys.stderr

# Отключаем вывод для импорта
sys.stdout = NullWriter()
sys.stderr = NullWriter()

import mlx.core as mx
from mlx_whisper import transcribe

# Возвращаем вывод
sys.stdout = original_stdout
sys.stderr = original_stderr

os.environ["MLX_WHISPER_VERBOSE"] = "0"
os.environ["TQDM_DISABLE"] = "1"

def transcribe_worker(args):
    """Функция для параллельной обработки"""
    idx, audio_path, language = args
    
    # Перехват вывода в процессе
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = NullWriter()
    sys.stderr = NullWriter()
    
    try:
        result = transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-medium-mlx",
            language=language,
            verbose=False,
            temperature=0.0,
        )
        text = result["text"].strip()
        error = None
    except Exception as e:
        text = ""
        error = str(e)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return idx, text, error
>>>>>>> 19d1af89d8e98c8af1b65e3266c0b7903aa13be9

def main():
    import argparse
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--language', default='ru')
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--save-interval', type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🔍 ПОИСК ФАЙЛОВ С ОШИБКОЙ")
    print(f"{'='*60}")

=======
    parser.add_argument('--language', default=None)
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

>>>>>>> 19d1af89d8e98c8af1b65e3266c0b7903aa13be9
    # Пути
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "raw_annotations.csv")
    audio_root = os.path.join(project_root, "data")
    
    # Загрузка CSV
<<<<<<< HEAD
    print(f"\n📂 Загрузка {csv_path}")
=======
>>>>>>> 19d1af89d8e98c8af1b65e3266c0b7903aa13be9
    df = pd.read_csv(csv_path)
    
    if "text" not in df.columns:
        df["text"] = ""
    
<<<<<<< HEAD
    # Сбор файлов с ошибкой
    files_to_process = []
    error_text = "[ERROR] Речь не распознана"
    
    for idx, row in df.iterrows():
        if args.max and len(files_to_process) >= args.max:
            break
        
        if str(row.get("text", "")) != error_text:
            continue
        
=======
    # Сбор файлов
    tasks = []
    for idx, row in df.iterrows():
        if args.max and len(tasks) >= args.max:
            break
        
>>>>>>> 19d1af89d8e98c8af1b65e3266c0b7903aa13be9
        rel_path = row.get("audio_path") or row.get("relative_path")
        if not isinstance(rel_path, str):
            continue
        
        audio_path = os.path.join(audio_root, rel_path.replace("\\", os.sep).replace("/", os.sep))
        
<<<<<<< HEAD
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
=======
        if not os.path.exists(audio_path):
            continue
        
        if not args.force and str(row.get("text", "")).strip():
            continue
        
        tasks.append((idx, audio_path, args.language))
    
    total_files = len(tasks)
    if total_files == 0:
        return
    
    print(f"Всего файлов: {total_files}")
    
    # Параллельная обработка
    results = {}
    errors = []
    
    with Pool(processes=args.workers) as pool:
        for i, (idx, text, error) in enumerate(pool.imap_unordered(transcribe_worker, tasks)):
            if error:
                errors.append(error)
            else:
                results[idx] = text
            
            # Счетчик + автосохранение каждые 50 файлов
            if (i + 1) % 50 == 0:
                print(f"Обработано: {i+1}")
                
                # АВТОСОХРАНЕНИЕ КАЖДЫЕ 50 ФАЙЛОВ
                for saved_idx, saved_text in results.items():
                    df.at[saved_idx, "text"] = saved_text
                df.to_csv(csv_path, index=False)
    
    # Финальное сохранение
    for idx, text in results.items():
        df.at[idx, "text"] = text
    df.to_csv(csv_path, index=False)
    
    print(f"\nГотово! Обработано: {len(results)}")
    if errors:
        print(f"Ошибок: {len(errors)}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
>>>>>>> 19d1af89d8e98c8af1b65e3266c0b7903aa13be9
    main()