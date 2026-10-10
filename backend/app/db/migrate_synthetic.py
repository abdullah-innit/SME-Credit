"""Non-destructive upgrade for the is_synthetic column on existing SQLite DBs."""

from sqlalchemy import Engine, inspect, text

from app.db.session import engine


def upgrade(bind: Engine = engine) -> bool:
    if bind.dialect.name != "sqlite":
        raise ValueError("This targeted upgrade supports SQLite only")
    with bind.begin() as connection:
        inspector = inspect(connection)
        if "businesses" not in inspector.get_table_names():
            return False
        if "is_synthetic" in {c["name"] for c in inspector.get_columns("businesses")}:
            return False
        connection.execute(
            text("ALTER TABLE businesses ADD COLUMN is_synthetic BOOLEAN NOT NULL DEFAULT 1")
        )
    return True


if __name__ == "__main__":
    changed = upgrade()
    print("Synthetic flag added; existing data preserved." if changed else "No upgrade needed.")
