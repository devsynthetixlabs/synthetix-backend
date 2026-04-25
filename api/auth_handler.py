from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from engine.auth_utils import decode_token

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        # credentials is an HTTPAuthorizationCredentials object
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            # .credentials is the actual STRING (the JWT)
            token_string = credentials.credentials 
            
            # Pass the STRING, not the whole object, to your decoder
            payload = decode_token(token_string) 
            
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            
            return payload