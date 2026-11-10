"""Unit tests for password hashing and token helpers in src/auth.py."""

import pytest

from src.auth import (
    DEMO_USERS,
    JWT_SECRET_ENV,
    ROLES,
    TokenError,
    authenticate,
    create_access_token,
    decode_access_token,
    get_jwt_secret,
)


def test_no_plaintext_passwords_stored():
    for username, user in DEMO_USERS.items():
        assert "password" not in user, username
        assert user["password_hash"].startswith("$2b$"), username
        assert "demo123" not in user["password_hash"], username


def test_every_demo_user_can_log_in_with_documented_password():
    for username, user in DEMO_USERS.items():
        profile = authenticate(username, "demo123")
        assert profile is not None, username
        assert profile["role"] == user["role"]
        assert profile["departments"] == ROLES[user["role"]]["departments"]
        assert "password_hash" not in profile


@pytest.mark.parametrize(
    "username,password",
    [("ceo", "wrong"), ("ceo", ""), ("ceo", "DEMO123"), ("nobody", "demo123"), ("", "")],
)
def test_bad_credentials_rejected(username, password):
    assert authenticate(username, password) is None


@pytest.mark.parametrize("secret", [None, "", "too-short"])
def test_missing_or_weak_secret_rejected(monkeypatch, secret):
    if secret is None:
        monkeypatch.delenv(JWT_SECRET_ENV, raising=False)
    else:
        monkeypatch.setenv(JWT_SECRET_ENV, secret)
    with pytest.raises(RuntimeError):
        get_jwt_secret()


def test_token_round_trip(monkeypatch):
    monkeypatch.setenv(JWT_SECRET_ENV, "x" * 32)
    user = authenticate("raj_eng", "demo123")
    claims = decode_access_token(create_access_token(user))
    assert claims["sub"] == "raj_eng"
    assert claims["role"] == "engineering"


def test_token_rejected_after_secret_rotation(monkeypatch):
    monkeypatch.setenv(JWT_SECRET_ENV, "a" * 32)
    token = create_access_token(authenticate("raj_eng", "demo123"))
    monkeypatch.setenv(JWT_SECRET_ENV, "b" * 32)
    with pytest.raises(TokenError):
        decode_access_token(token)
