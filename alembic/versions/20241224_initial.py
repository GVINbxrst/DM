"""create initial schema

Revision ID: 20241224_initial
Revises: 
Create Date: 2024-12-24
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20241224_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Создаем все таблицы из моделей
    from src.database.models import Base
    
    # Создаем metadata с движком
    bind = op.get_bind()
    Base.metadata.create_all(bind)

def downgrade() -> None:
    # Удаляем все таблицы
    from src.database.models import Base
    
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
