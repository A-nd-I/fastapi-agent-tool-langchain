from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import hashlib
import os

# Crear el directorio para la base de datos si no existe
os.makedirs('data', exist_ok=True)

# Si existe la base de datos, eliminarla
#if os.path.exists('data/users.db'):
#    os.remove('data/users.db')

# Configuración de la base de datos
SQLALCHEMY_DATABASE_URL = "sqlite:///data/users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    role = Column(String, default="user")
    
    # Campos para Stripe
    stripe_customer_id = Column(String, unique=True, nullable=True)
    subscription_status = Column(String, nullable=True)  # 'active', 'canceled', 'past_due', etc.
    subscription_plan = Column(String, nullable=True)    # 'basic', 'pro', etc.
    subscription_end_date = Column(DateTime, nullable=True)
    comparisons_count = Column(Integer, default=0)      # Contador de comparaciones realizadas
    last_comparison_date = Column(DateTime, nullable=True)

# Crear las tablas
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password against one provided by user"""
    return stored_password == hash_password(provided_password)

def create_user(db: SessionLocal, username: str, email: str, password: str):
    """Create a new user in the database"""
    try:
        db_user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password)
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise e

def get_user_by_username(db: SessionLocal, username: str):
    """Get a user by username"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: SessionLocal, email: str):
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first() 