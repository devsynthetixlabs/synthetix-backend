import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from engine.auth_utils import hash_password, verify_password, create_access_token, decode_token
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from api.auth_handler import JWTBearer
from engine.core import db

# Initialize the bearer
auth_guard = JWTBearer()

class UserSignup(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    tenant_id: str  # For now, we'll pass this manually; later it will be dynamicfrom sqlalchemy import text

class UserLogin(BaseModel):
    email: EmailStr
    password: str

router = APIRouter(
    prefix="/auth",  # All routes in this file will start with /auth
    tags=["Authentication"] # Organizes your Swagger docs
)

@router.post("/signup")
async def signup(user_data: UserSignup):
    # 1. Check if user already exists
    check_query = text("SELECT id FROM users WHERE email = :email")
    
    with db._engine.connect() as conn:
        existing_user = conn.execute(check_query, {"email": user_data.email}).fetchone()
        
        if existing_user:
            return {"error": "User with this email already exists"}, 400

        # 2. Hash the password
        hashed_pwd = hash_password(user_data.password)

        # 3. Insert the new user
        insert_query = text("""
            INSERT INTO users (email, password_hash, first_name, last_name, tenant_id)
            VALUES (:email, :pwd, :fn, :ln, :tid)
        """)
        
        try:
            conn.execute(insert_query, {
                "email": user_data.email,
                "pwd": hashed_pwd,
                "fn": user_data.first_name,
                "ln": user_data.last_name,
                "tid": user_data.tenant_id
            })
            conn.commit() # Important for persistent storage
            return {"message": "User created successfully", "email": user_data.email}
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}, 500

@router.post("/login") # This becomes /auth/login
async def login(credentials: UserLogin):
    # 1. Fetch user from DB
    query = text("""
        SELECT email, password_hash, tenant_id, first_name 
        FROM users 
        WHERE email = :email
    """)
    
    with db._engine.connect() as conn:
        user = conn.execute(query, {"email": credentials.email}).fetchone()

    # 2. Safety Check: Does user exist?
    if not user:
        return {"error": "Invalid email or password"}, 401

    # 3. Password Check
    # user[1] is the password_hash from our SQL query
    if not verify_password(credentials.password, user[1]):
        return {"error": "Invalid email or password"}, 401

    # 4. Generate the JWT
    # We embed the email (sub) and the tenant_id in the token payload
    token_data = {
        "sub": user[0],
        "tenant_id": user[2],
        "first_name": user[3]
    }
    
    access_token = create_access_token(data=token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "first_name": user[3],
            "email": user[0],
            "tenant_id": user[2]
        }
    }

def get_current_user(user: str = Depends(auth_guard)):
    """
    1. Extracts the token from the Authorization header.
    2. Decodes it using your SECRET_KEY.
    3. Returns the user data (email, tenant_id) if valid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # decode_token is the helper we wrote earlier to verify the JWT
    payload = user
    
    if payload is None:
        raise credentials_exception
        
    # This dictionary is what 'user' becomes in your @app.post("/ask") route
    return payload