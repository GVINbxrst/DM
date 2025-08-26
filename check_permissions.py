import asyncio
import os
import sys
sys.path.append('.')

# Устанавливаем переменную окружения для нашей БД
os.environ['DATABASE_URL'] = 'postgresql+asyncpg://diagmod_user:diagmod@localhost:5432/diagmod'
os.environ['APP_ENVIRONMENT'] = 'development'

async def grant_permissions():
    try:
        import asyncpg
        print("Предоставляю права пользователю diagmod_user...")
        
        # Подключаемся напрямую как diagmod_user
        conn = await asyncpg.connect(
            'postgresql://diagmod_user:diagmod@localhost:5432/diagmod'
        )
        
        # Проверяем текущие права
        result = await conn.fetchval(
            "SELECT has_schema_privilege('diagmod_user', 'public', 'CREATE')"
        )
        print(f"Права на CREATE в схеме public: {result}")
        
        result = await conn.fetchval(
            "SELECT has_database_privilege('diagmod_user', 'diagmod', 'CREATE')"
        )
        print(f"Права на CREATE в БД diagmod: {result}")
        
        await conn.close()
        print("✅ Проверка прав завершена")
        
    except Exception as e:
        print(f"❌ Ошибка проверки прав: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(grant_permissions())
