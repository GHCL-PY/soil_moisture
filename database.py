from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DB_URL = "postgresql://postgres:admin@localhost:5432/soil_moisture_db"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ALTER ROLE ghcl WITH CREATEDB;
# GRANT ALL PRIVILEGES ON SCHEMA public TO ghcl;
# ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ghcl;
