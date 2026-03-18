import pytest
from fastapi import HTTPException
from src.auth import verify_token


def test_verify_token_valid():
    verify_token("test-token")


def test_verify_token_invalid():
    with pytest.raises(HTTPException) as exc_info:
        verify_token("wrong-token")
    assert exc_info.value.status_code == 401


def test_verify_token_missing():
    with pytest.raises(HTTPException):
        verify_token(None)
