import urllib.request
import json

def test_api():
    try:
        # Проверка базового health
        response = urllib.request.urlopen('http://localhost:8000/health')
        health_data = json.loads(response.read().decode())
        print("✅ API Health:", health_data)
        
        # Проверка БД
        response = urllib.request.urlopen('http://localhost:8000/health/db')
        db_data = json.loads(response.read().decode())
        print("✅ DB Health:", db_data)
        
        # Загрузка CSV файла
        import urllib.parse
        
        with open('data/sample.csv', 'rb') as f:
            csv_content = f.read()
        
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="sample.csv"\r\n'
            f'Content-Type: text/csv\r\n'
            f'\r\n'
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
        print("✅ CSV Upload:", upload_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()
