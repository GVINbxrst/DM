import asyncio
import os
import sys
sys.path.append('.')

# Устанавливаем переменную окружения для нашей БД
os.environ['DATABASE_URL'] = 'postgresql+asyncpg://diagmod_user:diagmod@localhost:5432/diagmod'

async def test_connection():
    try:
        import asyncpg
        print("Тестируем прямое подключение asyncpg...")
        
        conn = await asyncpg.connect(
            'postgresql://diagmod_user:diagmod@localhost:5432/diagmod'
        )
        result = await conn.fetchval('SELECT 1')
        print(f"✅ Прямое подключение успешно: {result}")
        await conn.close()
        
        print("Тестируем подключение через SQLAlchemy...")
        import sqlalchemy
        from src.database.connection import engine
        
        async with engine.connect() as conn:
            result = await conn.execute(sqlalchemy.text('SELECT 1'))
            print(f"✅ SQLAlchemy подключение успешно: {result.scalar()}")
        
        print("✅ Все тесты подключения прошли успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
