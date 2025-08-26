import asyncio
import os
import pytest

pytestmark = pytest.mark.skipif(os.getenv("E2E_LOCAL") != "1", reason="External DB check; set E2E_LOCAL=1 to run")


async def test_db_connection_external():
    import asyncpg
    conn = await asyncpg.connect('postgresql://diagmod_user:diagmod@localhost:5432/diagmod')
    try:
        result = await conn.fetchval('SELECT 1')
        assert result == 1
    finally:
        await conn.close()
