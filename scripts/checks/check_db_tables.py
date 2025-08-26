import asyncio
import os
import sys
sys.path.append('.')

# Устанавливаем переменную окружения для нашей БД
os.environ['DATABASE_URL'] = 'postgresql+asyncpg://diagmod_user:diagmod@localhost:5432/diagmod'
os.environ['APP_ENVIRONMENT'] = 'development'

async def check_tables():
    try:
        print("Проверяю таблицы в БД...")
        import sqlalchemy
        from src.database.connection import engine
        
        async with engine.connect() as conn:
            # Проверяем список таблиц
            result = await conn.execute(sqlalchemy.text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ))
            tables = [row[0] for row in result]
            print(f"✅ Найдено таблиц: {len(tables)}")
            for table in sorted(tables):
                print(f"   - {table}")
            
            # Проверяем миграции
            if 'alembic_version' in tables:
                result = await conn.execute(sqlalchemy.text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                print(f"✅ Текущая версия миграций: {version}")
            else:
                print("❌ Таблица alembic_version не найдена - миграции не применены")
        
        print("✅ Проверка БД завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка проверки БД: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_tables())
