from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.config import get_settings

security = HTTPBearer()


def verify_token(token: str | None) -> None:
    if not token or token != get_settings().api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    verify_token(credentials.credentials)
    return credentials.credentials
