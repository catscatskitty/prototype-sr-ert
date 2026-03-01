"""
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default=None)
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    # Пути
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "raw_annotations.csv")
    audio_root = os.path.join(project_root, "data")
    
    # Загрузка CSV
    df = pd.read_csv(csv_path)
    
    if "text" not in df.columns:
        df["text"] = ""
    
    # Сбор файлов
    tasks = []
    for idx, row in df.iterrows():
        if args.max and len(tasks) >= args.max:
            break
        
        rel_path = row.get("audio_path") or row.get("relative_path")
        if not isinstance(rel_path, str):
            continue
        
        audio_path = os.path.join(audio_root, rel_path.replace("\\", os.sep).replace("/", os.sep))
        
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
    main()