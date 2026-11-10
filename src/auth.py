"""
Role-Based Access Control (RBAC) and authentication for FinSolve.

Shared by the Streamlit app and the FastAPI service, so both log users in
the same way and map roles to the same allowed departments.
"""

import os
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

# Role definitions with access permissions
ROLES = {
    "finance": {
        "name": "Finance Department",
        "departments": ["finance", "general"],
        "description": "Access to financial reports, budgets, and investment policies"
    },
    "hr": {
        "name": "Human Resources",
        "departments": ["hr", "general"],
        "description": "Access to HR policies, employee handbook, and leave policies"
    },
    "marketing": {
        "name": "Marketing Department",
        "departments": ["marketing", "general"],
        "description": "Access to marketing strategies, campaigns, and social media guidelines"
    },
    "engineering": {
        "name": "Engineering Department",
        "departments": ["engineering", "general"],
        "description": "Access to technical docs, architecture, and security protocols"
    },
    "executive": {
        "name": "C-Level Executive",
        "departments": ["finance", "hr", "marketing", "engineering", "executive", "general"],
        "description": "Full access to all company documents including board minutes"
    },
    "employee": {
        "name": "General Employee",
        "departments": ["general"],
        "description": "Access to general company policies and culture documents only"
    }
}

# Demo users. Passwords are stored as bcrypt hashes, never in plaintext.
# Every demo account's password is "demo123" (documented in the README);
# each hash has its own random salt, so equal passwords give different hashes.
# Regenerate a hash with:
#   python -c "import bcrypt; print(bcrypt.hashpw(b'NEW_PASSWORD', bcrypt.gensalt()).decode())"
DEMO_USERS = {
    "sarah_cfo": {
        "password_hash": "$2b$12$DR3Fb7paFcagGyVAkMBYN.Daj8MohOM/sD16OPKf8t943T6e9lnI2",
        "role": "finance",
        "name": "Sarah Chen (CFO)",
    },
    "mike_hr": {
        "password_hash": "$2b$12$qmAz4k66wVKthyvR6zJUY.6NpnQYEE/lDBIovQrknF/Z4BZxSng4i",
        "role": "hr",
        "name": "Mike Johnson (HR Director)",
    },
    "lisa_mkt": {
        "password_hash": "$2b$12$X9o1jl8jZ.wWFrGG8Vh7eOaT5VZh3BduN6GI72Rnj8R4346RAVZWC",
        "role": "marketing",
        "name": "Lisa Park (Marketing Lead)",
    },
    "raj_eng": {
        "password_hash": "$2b$12$iLpxmq23AG2gDyVcmL5FGeB..klqLNpHj39Ea/M3slGQH0w.T/iY2",
        "role": "engineering",
        "name": "Raj Patel (Tech Lead)",
    },
    "ceo": {
        "password_hash": "$2b$12$nkK58nsThgyaHIvvRIJ/KenHQy4KbNm3/dXFS1hah/4y37o2CgVmW",
        "role": "executive",
        "name": "Alex Thompson (CEO)",
    },
    "intern": {
        "password_hash": "$2b$12$KvEjZYjZ18XGFOkRKrPivOMQP7ublYHPKUsBj9zdN5/Y2J3l7Aike",
        "role": "employee",
        "name": "Jamie Lee (Intern)",
    },
}

# Hash of a random value nobody knows. Checked when the username does not
# exist, so a wrong username takes as long as a wrong password and response
# time does not reveal which usernames are real.
_DUMMY_HASH = b"$2b$12$VmPtYbYmbT4PsxV5gHpQ/uUUf3XKABxBcYdLi4oBpRCtyNb5rg4WC"


def authenticate(username: str, password: str) -> dict | None:
    """
    Check a username and password against the stored bcrypt hashes.

    Returns the user's profile (without the hash) or None.
    """
    user = DEMO_USERS.get(username)
    stored_hash = user["password_hash"].encode() if user else _DUMMY_HASH
    password_ok = bcrypt.checkpw(password.encode(), stored_hash)

    if user and password_ok:
        return {
            "username": username,
            "name": user["name"],
            "role": user["role"],
            "departments": ROLES[user["role"]]["departments"],
        }
    return None


# ---------- Signed tokens (used by the FastAPI service) ----------

JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 3600  # 1 hour
JWT_SECRET_ENV = "FINSOLVE_JWT_SECRET"


class TokenError(Exception):
    """The token is missing, malformed, expired, or not signed by us."""


def get_jwt_secret() -> str:
    """Read the signing secret from the environment; refuse weak or missing ones."""
    secret = os.getenv(JWT_SECRET_ENV, "")
    if len(secret) < 32:
        raise RuntimeError(
            f"{JWT_SECRET_ENV} must be set to a random string of at least 32 characters. "
            'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    return secret


def create_access_token(user: dict, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    """Issue a signed JWT that carries who the user is and which role they have."""
    now = datetime.now(UTC)
    claims = {
        "sub": user["username"],
        "role": user["role"],
        "name": user["name"],
        "iat": now,
        "exp": now + timedelta(seconds=ttl_seconds),
    }
    return jwt.encode(claims, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Verify a JWT and return its claims.

    Raises TokenError if the signature is wrong, the token has expired,
    a required claim is missing, or the role is not one we know.
    """
    try:
        claims = jwt.decode(
            token,
            get_jwt_secret(),
            # Pin the algorithm: never let the token header choose it.
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise TokenError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise TokenError("Invalid token") from e

    if claims["role"] not in ROLES:
        raise TokenError("Invalid token")
    return claims


def get_allowed_departments(role: str) -> list:
    """Get list of departments this role can access."""
    return ROLES.get(role, {}).get("departments", ["general"])


def can_access_document(role: str, doc_department: str) -> bool:
    """Check if a role can access a document from a specific department."""
    allowed = get_allowed_departments(role)
    return doc_department in allowed
