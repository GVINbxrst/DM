"""
Главный файл запуска Dashboard приложения
"""
import sys
import os
from pathlib import Path

# Добавляем корень проекта (/app/src) в PYTHONPATH, если его нет
src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Импортируем основное приложение (shell c сайдбаром)
from dashboard.shell import main

if __name__ == "__main__":
    main()
