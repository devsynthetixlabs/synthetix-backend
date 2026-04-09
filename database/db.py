from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Optimized Engine for 2026 Serverless Environments
engine = create_engine(
    DATABASE_URL,
    # Pool size 5 is perfect for Vercel Free Tier
    pool_size=5,
    max_overflow=10,
    # Recycles connections every hour to avoid "stale" errors
    pool_recycle=3600,
    # Essential for Neon/Postgres over SSL
    connect_args={"sslmode": "require"}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Utility to get a database session (Standard FastAPI Pattern)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()