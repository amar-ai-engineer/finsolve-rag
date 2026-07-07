"""
Role-Based Access Control (RBAC) for FinSolve.
"""

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

# Demo credentials (in production, use proper auth like JWT)
DEMO_USERS = {
    "sarah_cfo": {"password": "demo123", "role": "finance", "name": "Sarah Chen (CFO)"},
    "mike_hr": {"password": "demo123", "role": "hr", "name": "Mike Johnson (HR Director)"},
    "lisa_mkt": {"password": "demo123", "role": "marketing", "name": "Lisa Park (Marketing Lead)"},
    "raj_eng": {"password": "demo123", "role": "engineering", "name": "Raj Patel (Tech Lead)"},
    "ceo": {"password": "demo123", "role": "executive", "name": "Alex Thompson (CEO)"},
    "intern": {"password": "demo123", "role": "employee", "name": "Jamie Lee (Intern)"},
}


def authenticate(username: str, password: str) -> dict | None:
    """
    Simple demo authentication.
    """
    user = DEMO_USERS.get(username)
    if user and user["password"] == password:
        return {
            "username": username,
            "name": user["name"],
            "role": user["role"],
            "departments": ROLES[user["role"]]["departments"],
        }
    return None


def get_allowed_departments(role: str) -> list:
    """Get list of departments this role can access."""
    return ROLES.get(role, {}).get("departments", ["general"])


def can_access_document(role: str, doc_department: str) -> bool:
    """Check if a role can access a document from a specific department."""
    allowed = get_allowed_departments(role)
    return doc_department in allowed
