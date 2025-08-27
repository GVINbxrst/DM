# Инструкции по исправлению проблемы с подключением к БД

## Проблема
Роль `diagmod_user` создана, но у неё нет прав на логин.

Ошибка: `role "diagmod_user" is not permitted to log in`

## Решение через pgAdmin
1. Откройте pgAdmin
2. Подключитесь к серверу PostgreSQL
3. Перейдите в: Server -> PostgreSQL -> Login/Group Roles
4. Найдите роль `diagmod_user`
5. Кликните правой кнопкой -> Properties
6. На вкладке "General" поставьте галочку "Can login?"
7. Нажмите Save

## Альтернативное решение через SQL
Если есть доступ к PostgreSQL через другого пользователя с правами администратора:

```sql
ALTER ROLE diagmod_user WITH LOGIN;
GRANT CONNECT ON DATABASE diagmod TO diagmod_user;
GRANT USAGE ON SCHEMA public TO diagmod_user;
GRANT CREATE ON SCHEMA public TO diagmod_user;
```

## Проверка после исправления
После исправления запустите снова:
```
set E2E_LOCAL=1 && pytest -q tests/integration/test_db_connection_external.py
# Linux/macOS:
E2E_LOCAL=1 pytest -q tests/integration/test_db_connection_external.py
```

Если всё успешно, вы увидите:
- ✅ Прямое подключение успешно: 1
- ✅ SQLAlchemy подключение успешно: 1
- ✅ Все тесты подключения прошли успешно!
