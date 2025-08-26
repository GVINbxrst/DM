import asyncio
import asyncpg

async def check_tables():
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='diagmod_user',
            password='diagmod',
            database='diagmod'
        )
        
        # Проверим список таблиц
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        print("📋 Таблицы в базе данных:")
        for table in tables:
            print(f"  - {table['table_name']}")
            
        # Проверим содержимое после загрузки
        raw_signals = await conn.fetch("SELECT id, file_name, processing_status FROM raw_signals ORDER BY created_at DESC LIMIT 3")
        print(f"\n📊 RawSignals (последние 3):")
        for rs in raw_signals:
            print(f"  - {rs['id']}: {rs['file_name']} - статус: {rs['processing_status']}")
            
        await conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(check_tables())
