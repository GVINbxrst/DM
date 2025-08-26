import urllib.request
import json
import os
import pytest

# Эти тесты предполагают запущенный локальный стек (API на localhost:8000).
# По умолчанию пропускаем. Для запуска: E2E_LOCAL=1 pytest tests/integration -k api_smoke
pytestmark = pytest.mark.skipif(os.getenv("E2E_LOCAL") != "1", reason="Local smoke test; set E2E_LOCAL=1 to run")


def test_api_smoke():
    # Проверка базового health
    response = urllib.request.urlopen('http://localhost:8000/health')
    health_data = json.loads(response.read().decode())
    assert 'status' in health_data

    # Проверка БД
    response = urllib.request.urlopen('http://localhost:8000/health/db')
    db_data = json.loads(response.read().decode())
    assert 'database' in db_data

    # Загрузка CSV файла
    with open('data/sample.csv', 'rb') as f:
        csv_content = f.read()

    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="sample.csv"\r\n'
        f'Content-Type: text/csv\r\n\r\n'
        f'{csv_content.decode()}\r\n'
        f'--{boundary}--\r\n'
    ).encode()

    req = urllib.request.Request(
        'http://localhost:8000/api/v1/upload',
        data=body,
        method='POST'
    )
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

    response = urllib.request.urlopen(req)
    upload_data = json.loads(response.read().decode())
    assert 'rows_processed' in upload_data or 'status' in upload_data
