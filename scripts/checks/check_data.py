import asyncio
import os
import sys
sys.path.append('.')

# Устанавливаем переменную окружения для нашей БД
os.environ['DATABASE_URL'] = 'postgresql+asyncpg://diagmod_user:diagmod@localhost:5432/diagmod'
os.environ['APP_ENVIRONMENT'] = 'development'

async def check_data():
    try:
        print("Проверяю данные в БД...")
        import sqlalchemy
        from src.database.connection import engine
        
        async with engine.connect() as conn:
            # Проверяем raw_signals
            result = await conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM raw_signals"))
            raw_count = result.scalar()
            print(f"✅ Записей в raw_signals: {raw_count}")
            
            if raw_count > 0:
                result = await conn.execute(sqlalchemy.text("SELECT id, filename, equipment_id FROM raw_signals LIMIT 5"))
                rows = result.fetchall()
                print("   Примеры записей:")
                for row in rows:
                    print(f"   - ID: {row[0]}, File: {row[1]}, Equipment: {row[2]}")
            
            # Проверяем features
            result = await conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM features"))
            features_count = result.scalar()
            print(f"✅ Записей в features: {features_count}")
            
            if features_count > 0:
                result = await conn.execute(sqlalchemy.text("SELECT raw_signal_id, rms_r, rms_s, rms_t FROM features LIMIT 3"))
                rows = result.fetchall()
                print("   Примеры фич:")
                for row in rows:
                    print(f"   - Raw ID: {row[0]}, RMS R/S/T: {row[1]:.3f}/{row[2]:.3f}/{row[3]:.3f}")
        
        print("✅ Проверка данных завершена!")
        
    except Exception as e:
        print(f"❌ Ошибка проверки данных: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_data())
