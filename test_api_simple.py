import requests
import os

def test_api_simple():
    try:
        # Проверка health endpoint
        response = requests.get('http://localhost:8000/health')
        print("✅ API Health:", response.json())
        
        # Проверка health/db endpoint
        response = requests.get('http://localhost:8000/health/db')
        print("✅ DB Health:", response.json())
        
        # Загрузка CSV файла
        csv_file_path = 'data/sample.csv'
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'rb') as f:
                files = {'file': ('sample.csv', f, 'text/csv')}
                response = requests.post('http://localhost:8000/api/v1/upload', files=files)
                
            if response.status_code == 200:
                print("✅ CSV Upload successful:", response.json())
            else:
                print(f"❌ CSV Upload failed: {response.status_code}, {response.text}")
        else:
            print(f"❌ CSV файл не найден: {csv_file_path}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    test_api_simple()
