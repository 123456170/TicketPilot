import os
import re
import json
import time
import hashlib
import sqlite3
import tempfile
import textwrap
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None
    IMAGEIO_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

APP_NAME = "TicketPilot"
DB_PATH = Path(__file__).with_name("ticketpilot_lab.sqlite3")
DEMO_NOW = datetime(2026, 10, 19, 9, 30, 0, tzinfo=timezone.utc)

HIGH_RISK_CATEGORIES = {
    "password_reset",
    "account_unlock",
    "mfa_reset",
    "access_request",
}

SENSITIVE_KEYWORDS = [
    "password",
    "unlock",
    "mfa",
    "multi-factor",
    "authenticator",
    "access",
    "permission",
    "firewall",
    "enroll",
    "reset",
]

CATEGORY_KEYWORDS = {
    "password_reset": [
        "password", "passcode", "forgot password", "reset password",
        "password expired", "cannot sign in", "sign in password", "pwd"
    ],
    "account_unlock": [
        "locked", "unlock", "account lock", "lockout", "disabled account",
        "account is locked"
    ],
    "vpn": [
        "vpn", "tunnel", "split tunnel", "global protect", "anyconnect",
        "vpn profile", "vpn client"
    ],
    "wifi": [
        "wi-fi", "wifi", "wireless", "ssid", "802.1x", "wifi certificate",
        "wireless profile"
    ],
    "printer": [
        "printer", "print queue", "paper jam", "default printer",
        "offline printer", "spooler", "printing"
    ],
    "mfa_reset": [
        "mfa", "multi-factor", "authenticator", "otp token", "duo",
        "verification app", "mobile approve"
    ],
    "software_request": [
        "software", "install", "app request", "license request",
        "package", "deploy", "application request"
    ],
    "disk_space": [
        "disk space", "storage full", "cleanup", "temp files",
        "quota", "c: drive", "low disk"
    ],
    "email_sync": [
        "email sync", "outlook", "mail profile", "send receive",
        "ost", "exchange sync", "mailbox not syncing"
    ],
    "access_request": [
        "access", "permission", "sharepoint", "folder", "role",
        "entitlement", "mailbox access", "firewall rule", "lab access"
    ],
}

URGENCY_HIGH = [
    "urgent", "asap", "critical", "blocked", "production", "down",
    "cannot work", "emergency", "right now"
]
URGENCY_LOW = [
    "whenever", "no rush", "minor", "question", "info", "fyi"
]
SENTIMENT_NEGATIVE = [
    "frustrated", "angry", "annoyed", "broken", "fail", "not working",
    "still", "again", "worst", "unable", "cannot"
]
SENTIMENT_POSITIVE = [
    "thanks", "thank you", "great", "appreciate", "helpful", "awesome"
]

ACTION_MAP = {
    "password_reset": "reset_password",
    "account_unlock": "unlock_account",
    "mfa_reset": "reset_mfa",
    "vpn": "push_vpn_profile",
    "wifi": "push_wifi_profile",
    "printer": "set_default_printer",
    "software_request": "create_software_request",
    "disk_space": "recommend_disk_cleanup",
    "email_sync": "reset_mail_profile",
    "access_request": "grant_lab_access",
    "other": "provide_guidance",
}

USER_CHANGING_ACTIONS = {
    "reset_password",
    "unlock_account",
    "reset_mfa",
    "push_vpn_profile",
    "push_wifi_profile",
    "set_default_printer",
    "grant_lab_access",
}

ALLOWED_TICKET_FIELDS = {
    "category",
    "urgency",
    "sentiment",
    "confidence",
    "status",
    "resolution_summary",
    "runbook_slug",
    "runbook_title",
    "escalated",
    "escalation_reason",
    "duplicate_of",
    "verification_status",
    "action_status",
    "processed",
    "updated_at",
}


# =============================================================================
# Small utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_answer(label: str, answer: str) -> str:
    return sha256(f"ticketpilot|{str(label).strip().lower()}|{str(answer).strip().lower()}")


def mask_secret(value: Any, keep: int = 2) -> str:
    s = str(value)
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * max(3, len(s) - keep)


def mask_answer(answer: str) -> str:
    s = str(answer or "")
    if not s:
        return ""
    if "@" in s:
        local, domain = s.split("@", 1)
        return f"{local[:2]}***@{domain[:1]}***"
    return s[:2] + "***"


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def strip_code_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def random_created_at(rng: random.Random) -> str:
    return (DEMO_NOW - timedelta(minutes=rng.randint(5, 4320))).isoformat()


def row_to_dict(row: Any) -> Dict[str, Any]:
    return dict(row) if row is not None else {}


def fetch_one(db: sqlite3.Connection, query: str, params: Tuple = ()) -> Dict[str, Any]:
    cur = db.execute(query, params or ())
    return row_to_dict(cur.fetchone())


def fetch_all(db: sqlite3.Connection, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    cur = db.execute(query, params or ())
    return [dict(r) for r in cur.fetchall()]


# =============================================================================
# 25 runbooks, stored as markdown strings inside app.py
# =============================================================================

RUNBOOK_SPECS = [
    (
        "password-reset",
        "Password Reset",
        "password_reset",
        "high",
        "Reset a lab password after identity verification and issue a one-time token.",
        [
            "Confirm the requester exists in the mock directory.",
            "Ask two challenge questions before any reset.",
            "Generate a one-time token in the lab database only.",
            "Instruct the user to change the token at next sign-in.",
            "Write a hash-chained audit record.",
        ],
    ),
    (
        "account-unlock",
        "Account Unlock",
        "account_unlock",
        "high",
        "Unlock a locked lab account after verifying identity and confirming lock reason.",
        [
            "Check mock account lock state.",
            "Verify identity with two challenges.",
            "Remove the simulated lock flag.",
            "Advise the user to avoid repeated failed sign-ins.",
            "Audit the unlock decision.",
        ],
    ),
    (
        "mfa-reset",
        "MFA Reset",
        "mfa_reset",
        "high",
        "Reset multi-factor authentication enrollment for a verified lab user.",
        [
            "Confirm user identity with manager and employee ID challenges.",
            "Invalidate the old simulated MFA secret.",
            "Issue a re-enrollment marker in the lab directory.",
            "Send the user to the mock enrollment page.",
            "Record the MFA reset in the audit chain.",
        ],
    ),
    (
        "vpn-install-profile",
        "VPN Profile Install",
        "vpn",
        "medium",
        "Push a simulated VPN profile to the lab user record.",
        [
            "Confirm the user needs remote lab access.",
            "Push the lab VPN profile marker.",
            "Provide client installation guidance.",
            "Do not touch real network devices.",
            "Audit the profile push.",
        ],
    ),
    (
        "vpn-connectivity-troubleshoot",
        "VPN Connectivity Troubleshooting",
        "vpn",
        "medium",
        "Guide the user through safe VPN connectivity checks in the lab.",
        [
            "Ask the user to confirm client version.",
            "Check simulated profile presence.",
            "Recommend reconnect and DNS flush guidance.",
            "Escalate only if the issue persists.",
            "Log the guidance outcome.",
        ],
    ),
    (
        "wifi-join-profile",
        "Wi-Fi Profile Join",
        "wifi",
        "medium",
        "Provide steps for joining the lab Wi-Fi profile.",
        [
            "Confirm device is in the lab inventory.",
            "Push or verify the simulated Wi-Fi profile marker.",
            "Share SSID and connection steps.",
            "Avoid real wireless controller changes.",
            "Audit the action.",
        ],
    ),
    (
        "wifi-cert-renewal",
        "Wi-Fi Certificate Renewal",
        "wifi",
        "medium",
        "Renew a simulated machine certificate for lab Wi-Fi access.",
        [
            "Check certificate expiry in the mock record.",
            "Issue a renewal marker.",
            "Instruct the user to reconnect.",
            "Never contact a real CA.",
            "Write audit evidence.",
        ],
    ),
    (
        "printer-add-default",
        "Add Default Printer",
        "printer",
        "low",
        "Set the default lab printer in the mock user profile.",
        [
            "Identify the user location from the directory.",
            "Set the simulated default printer queue.",
            "Provide add-printer steps.",
            "Do not access real print servers.",
            "Audit the change.",
        ],
    ),
    (
        "printer-queue-clear",
        "Clear Printer Queue",
        "printer",
        "low",
        "Guide the user to clear a stuck local print queue safely.",
        [
            "Ask the user to pause printing.",
            "Recommend local queue clear steps.",
            "Avoid remote spooler commands.",
            "Confirm test page guidance.",
            "Log the resolution.",
        ],
    ),
    (
        "printer-offline-network",
        "Printer Offline Network Check",
        "printer",
        "low",
        "Troubleshoot simulated printer offline status with non-invasive checks.",
        [
            "Verify printer name in the lab record.",
            "Recommend network reachability checks by the user.",
            "Escalate only if multiple users are affected.",
            "Do not ping or block real hosts.",
            "Audit the decision.",
        ],
    ),
    (
        "software-request-approval",
        "Software Request Approval",
        "software_request",
        "medium",
        "Create a simulated software request record for lab approval.",
        [
            "Capture software name and business reason.",
            "Create a mock request ID.",
            "Route to the simulated approval queue.",
            "Never install real software.",
            "Write audit evidence.",
        ],
    ),
    (
        "software-install-failed",
        "Software Install Failed",
        "software_request",
        "low",
        "Provide safe troubleshooting for a failed lab software request.",
        [
            "Check request status in the mock system.",
            "Recommend retry or package repair guidance.",
            "Collect error text from the user.",
            "Escalate if the package is unavailable.",
            "Audit the outcome.",
        ],
    ),
    (
        "disk-space-cleanup",
        "Disk Space Cleanup",
        "disk_space",
        "low",
        "Recommend safe disk cleanup steps without running OS commands.",
        [
            "Identify reported drive and usage.",
            "Suggest temp file and recycle bin cleanup.",
            "Recommend moving large files to lab storage.",
            "Never execute remote cleanup commands.",
            "Log guidance provided.",
        ],
    ),
    (
        "email-sync-outlook",
        "Email Sync Outlook",
        "email_sync",
        "low",
        "Resolve common Outlook mail sync issues with guided steps.",
        [
            "Check online status and profile guidance.",
            "Recommend send/receive and restart steps.",
            "Avoid resetting real mailboxes.",
            "Escalate if the mailbox is server-side blocked.",
            "Audit the resolution.",
        ],
    ),
    (
        "mailbox-permissions",
        "Mailbox Permissions",
        "access_request",
        "medium",
        "Handle simulated mailbox permission requests with identity verification.",
        [
            "Verify requester identity.",
            "Confirm owner approval in the mock workflow.",
            "Grant a simulated permission marker only.",
            "Never modify real mailbox ACLs.",
            "Write hash-chained audit evidence.",
        ],
    ),
    (
        "shared-mailbox-access",
        "Shared Mailbox Access",
        "access_request",
        "medium",
        "Provision simulated shared mailbox access after approval.",
        [
            "Validate business need.",
            "Check manager approval in the lab record.",
            "Add a mock membership marker.",
            "Do not touch Exchange production.",
            "Audit the access grant.",
        ],
    ),
    (
        "sso-signin-loop",
        "SSO Sign-in Loop",
        "password_reset",
        "medium",
        "Break simulated single sign-on loops with safe user-side guidance.",
        [
            "Ask the user to clear browser session cookies.",
            "Verify account state in the mock directory.",
            "Recommend private window testing.",
            "Escalate if identity provider simulation is down.",
            "Log the steps.",
        ],
    ),
    (
        "office-license-assignment",
        "Office License Assignment",
        "software_request",
        "low",
        "Assign a simulated office license marker to the lab user.",
        [
            "Confirm user eligibility.",
            "Set the mock license field.",
            "Provide sign-out/sign-in guidance.",
            "Avoid real tenant changes.",
            "Audit the assignment.",
        ],
    ),
    (
        "teams-device-permission",
        "Teams Device Permission",
        "software_request",
        "low",
        "Guide the user to enable microphone and camera permissions locally.",
        [
            "Provide OS-level permission steps.",
            "Recommend Teams client restart.",
            "Do not change real policy assignments.",
            "Escalate only if policy-blocked.",
            "Record guidance.",
        ],
    ),
    (
        "zoom-audio-driver",
        "Zoom Audio Driver",
        "software_request",
        "low",
        "Troubleshoot simulated audio driver issues with user-safe steps.",
        [
            "Ask the user to select the correct device.",
            "Recommend driver update guidance from the vendor site.",
            "Avoid remote driver installation.",
            "Escalate if hardware failure is suspected.",
            "Audit the outcome.",
        ],
    ),
    (
        "browser-cache-policies",
        "Browser Cache and Policies",
        "software_request",
        "low",
        "Clear local browser cache and validate simulated policy state.",
        [
            "Recommend cache clear steps.",
            "Check mock browser policy marker.",
            "Avoid touching real endpoint management.",
            "Escalate if policy conflict persists.",
            "Log the resolution.",
        ],
    ),
    (
        "proxy-pac-file",
        "Proxy PAC File",
        "access_request",
        "medium",
        "Validate simulated proxy auto-config assignment for lab browsing.",
        [
            "Confirm user network zone in the mock record.",
            "Push a simulated PAC profile marker.",
            "Provide browser proxy check guidance.",
            "Never edit real proxy servers.",
            "Audit the action.",
        ],
    ),
    (
        "firewall-application-rule",
        "Firewall Application Rule",
        "access_request",
        "high",
        "Simulate an application firewall rule request with strict approval.",
        [
            "Verify identity and business justification.",
            "Create a mock rule request only.",
            "Require human approval before any executed state change.",
            "Never block or allow real traffic.",
            "Write detailed audit evidence.",
        ],
    ),
    (
        "access-review-certification",
        "Access Review Certification",
        "access_request",
        "high",
        "Handle simulated access certification tasks with manager approval.",
        [
            "Identify the certification campaign in the lab.",
            "Verify requester and approver.",
            "Mark the mock certification complete.",
            "Do not alter real entitlement systems.",
            "Audit every decision.",
        ],
    ),
    (
        "mobile-device-enrollment",
        "Mobile Device Enrollment",
        "access_request",
        "high",
        "Guide simulated mobile enrollment without contacting real MDM.",
        [
            "Verify user identity.",
            "Issue a mock enrollment invitation marker.",
            "Provide QR/enrollment steps.",
            "Never wipe or lock a real device.",
            "Record the enrollment simulation.",
        ],
    ),
]


def build_runbooks() -> List[Dict[str, Any]]:
    out = []
    for slug, title, category, risk, summary, steps in RUNBOOK_SPECS:
        step_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        markdown = (
            f"# {title}\n\n"
            f"**Category:** `{category}`  \n"
            f"**Risk:** `{risk}`  \n"
            f"**Slug:** `{slug}`\n\n"
            f"## Summary\n{summary}\n\n"
            f"## Steps\n{step_text}\n\n"
            f"## Safety\n"
            f"This runbook is executed only against the local synthetic lab. "
            f"Dry-run is the default. Consequential actions require an explicit human approval record.\n"
        )
        search_text = f"{title} {category} {risk} {summary} {' '.join(steps)} {slug}".lower()
        out.append(
            {
                "slug": slug,
                "title": title,
                "category": category,
                "risk": risk,
                "summary": summary,
                "steps": steps,
                "markdown": markdown,
                "search_text": search_text,
            }
        )
    return out


RUNBOOKS = build_runbooks()
RUNBOOK_BY_SLUG = {r["slug"]: r for r in RUNBOOKS}


# =============================================================================
# Synthetic mock directory
# =============================================================================

FIRST_NAMES = [
    "Ava", "Liam", "Maya", "Noah", "Zoe", "Ethan", "Ivy", "Lucas", "Nora", "Owen",
    "Ruby", "Adam", "Ella", "Felix", "Grace", "Henry", "Iris", "Jack", "Kira", "Leo"
]
LAST_NAMES = [
    "Anderson", "Brown", "Clark", "Davis", "Evans", "Ford", "Garcia", "Harris",
    "Islam", "Jones", "Kumar", "Lee", "Martinez", "Nguyen", "Olson", "Patel",
    "Quinn", "Rivera", "Smith", "Turner"
]
DEPARTMENTS = [
    "Finance Lab", "Engineering Lab", "HR Lab", "Operations Lab", "Security Lab",
    "Sales Lab", "Legal Lab", "IT Lab", "Research Lab", "Support Lab"
]


def build_mock_users() -> List[Dict[str, Any]]:
    users = []
    for i in range(30):
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[(i * 7 + 3) % len(LAST_NAMES)]
        emp = f"E{1001 + i}"
        email = f"{first.lower()}.{last.lower()}{i}@lab.example"
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        manager_email = f"manager.{dept.split()[0].lower()}@lab.example"
        printer = f"PRN-{dept.split()[0][:3].upper()}-{(i % 5) + 1}"

        if i % 2 == 0:
            q1_label = "Manager email"
            q1_answer = manager_email
        else:
            q1_label = "Department"
            q1_answer = dept

        if i % 3 == 0:
            q2_label = "Preferred printer"
            q2_answer = printer
        else:
            q2_label = "Employee ID last 4"
            q2_answer = emp[-4:]

        users.append(
            {
                "employee_id": emp,
                "name": f"{first} {last}",
                "email": email,
                "department": dept,
                "manager_email": manager_email,
                "printer_queue": printer,
                "account_locked": 1 if i % 9 == 0 else 0,
                "password_token": None,
                "mfa_secret_masked": "MFA-******",
                "vpn_profile": None,
                "wifi_profile": None,
                "access_profile": None,
                "q1_label": q1_label,
                "q1_answer": q1_answer,
                "q2_label": q2_label,
                "q2_answer": q2_answer,
            }
        )
    return users


MOCK_USERS = build_mock_users()


def demo_answer_for_label(user: Dict[str, Any], label: str) -> str:
    lab = str(label).lower()
    if "manager" in lab:
        return str(user.get("manager_email", ""))
    if "department" in lab:
        return str(user.get("department", ""))
    if "employee id" in lab or "last 4" in lab:
        return str(user.get("employee_id", ""))[-4:]
    if "printer" in lab:
        return str(user.get("printer_queue", ""))
    return ""


# =============================================================================
# 100 synthetic tickets
# =============================================================================

NORMAL_COUNTS = {
    "password_reset": 22,
    "account_unlock": 9,
    "vpn": 13,
    "wifi": 7,
    "printer": 13,
    "mfa_reset": 7,
    "software_request": 6,
    "disk_space": 5,
    "email_sync": 8,
    "access_request": 5,
}

SUBJECT_TEMPLATES = {
    "password_reset": [
        "Password reset request",
        "Cannot sign in after password expiry",
        "Forgot my lab password",
    ],
    "account_unlock": [
        "Account locked after failed sign-ins",
        "Please unlock my account",
        "Lockout notice this morning",
    ],
    "vpn": [
        "VPN not connecting",
        "Need VPN profile for lab access",
        "VPN tunnel drops repeatedly",
    ],
    "wifi": [
        "Wi-Fi profile missing",
        "Cannot join lab wireless",
        "Wi-Fi certificate expired",
    ],
    "printer": [
        "Default printer missing",
        "Print queue stuck",
        "Printer shows offline",
    ],
    "mfa_reset": [
        "Authenticator app reset",
        "MFA token not arriving",
        "Need MFA re-enrollment",
    ],
    "software_request": [
        "Software request for lab tool",
        "Install failed for requested app",
        "License request for analysis tool",
    ],
    "disk_space": [
        "Disk space low warning",
        "Cleanup needed on C drive",
        "Storage quota almost full",
    ],
    "email_sync": [
        "Outlook not syncing",
        "Mail send receive failing",
        "Email profile needs reset",
    ],
    "access_request": [
        "Request access to finance share",
        "Need folder permission",
        "Firewall rule request for lab app",
    ],
}

BODY_TEMPLATES = {
    "password_reset": [
        "I forgot my password and need a reset to access lab email.",
        "My password expired and I cannot sign in. Please reset it.",
        "I need a password reset for my account. I can verify with my manager.",
        "Urgent: password not working and I am blocked from the lab portal.",
    ],
    "account_unlock": [
        "My account is locked after several failed attempts. Please unlock it.",
        "I got a lockout message and cannot sign in. Need account unlock.",
        "Please unlock my account, I confirmed my identity with my manager.",
    ],
    "vpn": [
        "The VPN client cannot establish a tunnel to the lab network.",
        "I need the lab VPN profile installed on my laptop.",
        "VPN connects for a minute and then drops. Please help.",
    ],
    "wifi": [
        "My laptop does not show the lab Wi-Fi profile.",
        "I cannot join the wireless network after the certificate expired.",
        "Please help me connect to the lab Wi-Fi SSID.",
    ],
    "printer": [
        "My default printer disappeared from the laptop.",
        "The print queue is stuck and jobs will not clear.",
        "The lab printer shows offline and I cannot print.",
    ],
    "mfa_reset": [
        "I replaced my phone and need MFA reset for the authenticator app.",
        "The MFA token is not arriving and I cannot complete sign-in.",
        "Please reset my multi-factor authentication enrollment.",
    ],
    "software_request": [
        "I need a software request approved for the lab analysis tool.",
        "The software install failed after the request was approved.",
        "Please assign a license for the requested application.",
    ],
    "disk_space": [
        "My disk space is low and I cannot save lab files.",
        "Please advise cleanup steps for the C drive.",
        "Storage quota is almost full on my workstation.",
    ],
    "email_sync": [
        "Outlook is not syncing new mail from the lab mailbox.",
        "Send and receive keeps failing in Outlook.",
        "I think my mail profile needs a reset.",
    ],
    "access_request": [
        "I need access to the finance share for the monthly lab report.",
        "Please grant folder permission to the engineering lab directory.",
        "I need a firewall rule request reviewed for a lab application.",
    ],
}


def generate_ticket_specs() -> List[Dict[str, Any]]:
    rng = random.Random(20261019)
    specs: List[Dict[str, Any]] = []

    for cat, n in NORMAL_COUNTS.items():
        for _ in range(n):
            user = rng.choice(MOCK_USERS)
            subject = rng.choice(SUBJECT_TEMPLATES[cat])
            body = rng.choice(BODY_TEMPLATES[cat])
            specs.append(
                {
                    "true_category": cat,
                    "subject": subject,
                    "body": body,
                    "user_email": user["email"],
                    "created_at": random_created_at(rng),
                    "demo_flag": "",
                }
            )

    # Three duplicate tickets, same user and subject as earlier tickets.
    for orig in specs[:3]:
        dup = dict(orig)
        dup["demo_flag"] = "duplicate"
        try:
            base = datetime.fromisoformat(orig["created_at"])
        except Exception:
            base = DEMO_NOW
        dup["created_at"] = (base + timedelta(minutes=rng.randint(3, 45))).isoformat()
        specs.append(dup)

    # One intentionally ambiguous low-confidence ticket.
    user = rng.choice(MOCK_USERS)
    specs.append(
        {
            "true_category": "other",
            "subject": "Help with my thing",
            "body": "Something is broken with my thing and I need help. It is urgent but I do not know what is failing.",
            "user_email": user["email"],
            "created_at": random_created_at(rng),
            "demo_flag": "low_confidence",
        }
    )

    # One sensitive ticket that will fail demo identity verification.
    user = rng.choice(MOCK_USERS)
    specs.append(
        {
            "true_category": "password_reset",
            "subject": "Password reset request",
            "body": "I forgot my password and need a reset. I cannot answer the security questions correctly.",
            "user_email": user["email"],
            "created_at": random_created_at(rng),
            "demo_flag": "wrong_verification",
        }
    )

    while len(specs) < 100:
        user = MOCK_USERS[len(specs) % len(MOCK_USERS)]
        specs.append(
            {
                "true_category": "password_reset",
                "subject": "Password reset request",
                "body": "I need a password reset for my lab account.",
                "user_email": user["email"],
                "created_at": random_created_at(rng),
                "demo_flag": "",
            }
        )

    return specs[:100]


TICKET_SPECS = generate_ticket_specs()


# =============================================================================
# SQLite lab database and hash-chained audit trail
# =============================================================================

def init_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            department TEXT,
            manager_email TEXT,
            account_locked INTEGER DEFAULT 0,
            password_token TEXT,
            mfa_secret_masked TEXT,
            vpn_profile TEXT,
            wifi_profile TEXT,
            printer_queue TEXT,
            access_profile TEXT,
            q1_label TEXT,
            q1_answer_hash TEXT,
            q2_label TEXT,
            q2_answer_hash TEXT
        );

        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            updated_at TEXT,
            user_email TEXT,
            subject TEXT,
            body TEXT,
            true_category TEXT,
            category TEXT,
            urgency TEXT,
            sentiment TEXT,
            confidence REAL,
            status TEXT DEFAULT 'Open',
            resolution_summary TEXT,
            runbook_slug TEXT,
            runbook_title TEXT,
            escalated INTEGER DEFAULT 0,
            escalation_reason TEXT,
            duplicate_of INTEGER DEFAULT 0,
            verification_status TEXT,
            action_status TEXT,
            demo_flag TEXT,
            processed INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT,
            status TEXT NOT NULL,
            details TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL
        );
        """
    )
    db.commit()


def seed_users(db: sqlite3.Connection) -> None:
    count = fetch_one(db, "SELECT COUNT(*) AS c FROM users")["c"]
    if count > 0:
        return

    for u in MOCK_USERS:
        db.execute(
            """
            INSERT INTO users (
                employee_id, name, email, department, manager_email,
                account_locked, password_token, mfa_secret_masked,
                vpn_profile, wifi_profile, printer_queue, access_profile,
                q1_label, q1_answer_hash, q2_label, q2_answer_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                u["employee_id"],
                u["name"],
                u["email"],
                u["department"],
                u["manager_email"],
                u["account_locked"],
                u["password_token"],
                u["mfa_secret_masked"],
                u["vpn_profile"],
                u["wifi_profile"],
                u["printer_queue"],
                u["access_profile"],
                u["q1_label"],
                hash_answer(u["q1_label"], u["q1_answer"]),
                u["q2_label"],
                hash_answer(u["q2_label"], u["q2_answer"]),
            ),
        )
    db.commit()


def seed_tickets(db: sqlite3.Connection) -> None:
    count = fetch_one(db, "SELECT COUNT(*) AS c FROM tickets")["c"]
    if count > 0:
        return

    for spec in TICKET_SPECS:
        db.execute(
            """
            INSERT INTO tickets (
                created_at, updated_at, user_email, subject, body,
                true_category, status, demo_flag, processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                spec["created_at"],
                spec["created_at"],
                spec["user_email"],
                spec["subject"],
                spec["body"],
                spec["true_category"],
                "Open",
                spec.get("demo_flag", ""),
            ),
        )
    db.commit()


def get_db_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    seed_users(conn)
    seed_tickets(conn)
    return conn


def reset_db(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        DROP TABLE IF EXISTS tickets;
        DROP TABLE IF EXISTS audit;
        DROP TABLE IF EXISTS users;
        """
    )
    init_schema(db)
    seed_users(db)
    seed_tickets(db)
    db.commit()


def compute_audit_hash(prev_hash: str, ts: str, actor: str, action: str, target: str, status: str, details_json: str) -> str:
    return sha256(f"{prev_hash}|{ts}|{actor}|{action}|{target}|{status}|{details_json}")


def append_audit(
    db: sqlite3.Connection,
    actor: str,
    action: str,
    target: str,
    status: str,
    details: Dict[str, Any],
) -> str:
    last = fetch_one(db, "SELECT entry_hash FROM audit ORDER BY id DESC LIMIT 1")
    prev_hash = last.get("entry_hash") if last else "GENESIS"
    ts = now_iso()
    details_json = json_dumps(details)
    entry_hash = compute_audit_hash(prev_hash, ts, actor, action, target, status, details_json)
    db.execute(
        """
        INSERT INTO audit (ts, actor, action, target, status, details, prev_hash, entry_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, actor, action, target, status, details_json, prev_hash, entry_hash),
    )
    db.commit()
    return entry_hash


def verify_audit_chain(db: sqlite3.Connection) -> Tuple[bool, int, str]:
    rows = fetch_all(db, "SELECT * FROM audit ORDER BY id")
    prev_hash = "GENESIS"
    for i, r in enumerate(rows, start=1):
        expected = compute_audit_hash(
            prev_hash,
            r["ts"],
            r["actor"],
            r["action"],
            r["target"] or "",
            r["status"],
            r["details"],
        )
        if r["prev_hash"] != prev_hash or r["entry_hash"] != expected:
            return False, i, prev_hash
        prev_hash = r["entry_hash"]
    return True, len(rows), prev_hash


def audit_to_json(db: sqlite3.Connection) -> bytes:
    rows = fetch_all(db, "SELECT * FROM audit ORDER BY id")
    return json.dumps(rows, indent=2, default=str).encode("utf-8")


def update_ticket(db: sqlite3.Connection, ticket_id: int, **fields: Any) -> None:
    clean = {k: v for k, v in fields.items() if k in ALLOWED_TICKET_FIELDS}
    clean["updated_at"] = now_iso()
    if not clean:
        return
    cols = ", ".join(f"{k} = ?" for k in clean.keys())
    vals = list(clean.values()) + [ticket_id]
    db.execute(f"UPDATE tickets SET {cols} WHERE id = ?", vals)
    db.commit()


# =============================================================================
# TF-IDF retriever and deterministic classifier
# =============================================================================

_RETRIEVER_CACHE: Optional[Tuple[TfidfVectorizer, Any]] = None


def get_retriever() -> Tuple[TfidfVectorizer, Any]:
    global _RETRIEVER_CACHE
    if _RETRIEVER_CACHE is None:
        docs = [r["search_text"] for r in RUNBOOKS]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(docs)
        _RETRIEVER_CACHE = (vec, mat)
    return _RETRIEVER_CACHE


def retrieve_runbook(query: str) -> Dict[str, Any]:
    query = str(query or "").strip().lower()
    if not query:
        return {**RUNBOOKS[0], "score": 0.0}
    vec, mat = get_retriever()
    qvec = vec.transform([query])
    sims = cosine_similarity(qvec, mat).flatten()
    idx = int(np.argmax(sims))
    return {**RUNBOOKS[idx], "score": float(sims[idx])}


def retrieve_top_n(query: str, n: int = 5) -> List[Dict[str, Any]]:
    query = str(query or "").strip().lower()
    if not query:
        return []
    vec, mat = get_retriever()
    qvec = vec.transform([query])
    sims = cosine_similarity(qvec, mat).flatten()
    order = np.argsort(sims)[::-1][:n]
    return [{**RUNBOOKS[int(i)], "score": float(sims[int(i)])} for i in order]


def classify_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()

    scores: Dict[str, float] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        s = 0.0
        for kw in kws:
            if kw in text:
                s += 1.0 + (0.5 if " " in kw else 0.0)
        if s > 0:
            scores[cat] = s

    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        category, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = 0.60 + 0.06 * best
        if best - second <= 0.5:
            confidence -= 0.10
        if any(w in text for w in URGENCY_HIGH):
            confidence += 0.02
        confidence = max(0.35, min(0.95, confidence))
    else:
        rb = retrieve_runbook(text)
        if rb["score"] > 0.12:
            category = rb["category"]
            confidence = min(0.69, 0.35 + 0.25 * rb["score"])
        else:
            category = "other"
            confidence = 0.35

    if ticket.get("demo_flag") == "low_confidence":
        confidence = min(confidence, 0.45)

    if any(w in text for w in URGENCY_HIGH):
        urgency = "high"
    elif any(w in text for w in URGENCY_LOW):
        urgency = "low"
    else:
        urgency = "medium"

    neg = sum(1 for w in SENTIMENT_NEGATIVE if w in text)
    pos = sum(1 for w in SENTIMENT_POSITIVE if w in text)
    if neg > pos:
        sentiment = "negative"
    elif pos > neg:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
        "confidence": float(round(confidence, 3)),
    }


def quick_requires_identity(ticket: Dict[str, Any]) -> bool:
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()
    return any(kw in text for kw in SENSITIVE_KEYWORDS)


def find_duplicate(db: sqlite3.Connection, ticket: Dict[str, Any]) -> int:
    row = fetch_one(
        db,
        "SELECT id FROM tickets WHERE user_email = ? AND subject = ? AND id <> ? ORDER BY id LIMIT 1",
        (ticket.get("user_email"), ticket.get("subject"), ticket.get("id")),
    )
    return int(row["id"]) if row else 0


# =============================================================================
# Event model, approval gate, schema validation, agents, crew
# =============================================================================

def make_event(agent: str, level: str, msg: str, kpi: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ts": now_iso(),
        "agent": agent,
        "level": level if level in {"info", "ok", "warn", "crit"} else "info",
        "msg": str(msg),
        "kpi": kpi or {},
    }


def emit_event(state: Dict[str, Any], agent: str, level: str, msg: str, kpi: Optional[Dict[str, Any]] = None) -> None:
    ev = make_event(agent, level, msg, kpi)
    log = state.get("event_log")
    if isinstance(log, list):
        log.append(ev)
        if len(log) > 5000:
            del log[:-3000]


def approval_gate(action: Dict[str, Any], approved: bool = False, dry_run: bool = True) -> Dict[str, Any]:
    consequential = bool(action.get("consequential", True))
    if dry_run:
        status = "dry_run"
    elif consequential and not approved:
        status = "pending_approval"
    else:
        status = "executed"
    return {
        "status": status,
        "action": action,
        "approved": bool(approved),
        "dry_run": bool(dry_run),
        "ts": now_iso(),
    }


def validate_schema(data: Any, schema: Dict[str, type]) -> Tuple[bool, Dict[str, Any]]:
    if not isinstance(data, dict):
        return False, {}
    if not schema:
        return True, data

    out: Dict[str, Any] = {}
    for key, typ in schema.items():
        if key not in data:
            return False, {}
        val = data[key]
        if typ is bool:
            if not isinstance(val, bool):
                return False, {}
        elif typ is int:
            if isinstance(val, bool) or not isinstance(val, int):
                return False, {}
        elif typ is float:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return False, {}
            val = float(val)
        else:
            if not isinstance(val, typ):
                return False, {}
        out[key] = val
    return True, out


def safe_state_for_model(state: Dict[str, Any]) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    user = state.get("user", {}) or {}
    public_user = {
        k: user.get(k)
        for k in ("employee_id", "name", "email", "department", "manager_email", "printer_queue")
        if k in user
    }
    public_ticket = {
        "id": ticket.get("id"),
        "subject": str(ticket.get("subject", ""))[:500],
        "body": str(ticket.get("body", ""))[:1500],
        "user_email": ticket.get("user_email"),
        "demo_flag": ticket.get("demo_flag"),
    }
    return {
        "ticket": public_ticket,
        "user": public_user,
        "intake": state.get("intake"),
        "classifier": state.get("classifier"),
        "retriever": {k: state.get("retriever", {}).get(k) for k in ("slug", "title", "category", "score")} if state.get("retriever") else None,
        "verifier": state.get("verifier"),
        "critic": state.get("critic"),
        "scenario": state.get("scenario"),
        "metrics": state.get("metrics"),
    }


class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        demo_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        schema: Dict[str, type],
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.demo_fn = demo_fn
        self.schema = schema

    def _call_model(self, api_key: str, task: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            from google import generativeai as genai  # type: ignore
        except Exception:
            return None

        state = context.get("state", {}) or {}
        payload = safe_state_for_model(state)
        prompt = (
            f"Task: {task}\n"
            f"Return valid JSON only matching the agent schema.\n"
            f"Context:\n{json_dumps(payload)[:8000]}"
        )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=self.system_prompt)

        try:
            config = genai.GenerationConfig(temperature=0.2, response_mime_type="application/json")
            response = model.generate_content(prompt, generation_config=config)
        except Exception:
            response = model.generate_content(prompt)

        text = getattr(response, "text", "") or ""
        text = strip_code_fences(text)
        if not text:
            raise ValueError("empty model response")
        return json.loads(text)

    def _fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            state = context.get("state", {}) or {}
            raw = self.demo_fn(state, context)
            ok, val = validate_schema(raw, self.schema)
            if ok:
                return val
        except Exception:
            pass
        return {"error": "model unavailable"}

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        json_mode: bool = True,
        api_key: Optional[str] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        for attempt in range(max(1, retries)):
            try:
                if api_key:
                    raw = self._call_model(api_key, task, context)
                    if raw is None:
                        return self._fallback(context)
                else:
                    state = context.get("state", {}) or {}
                    raw = self.demo_fn(state, context)

                ok, val = validate_schema(raw, self.schema)
                if ok:
                    return val
            except Exception:
                pass

            if attempt < retries - 1:
                time.sleep((2 ** attempt) * 0.05)

        return self._fallback(context)


class Crew:
    def __init__(self, agents: List[Agent], on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.agents = {a.name: a for a in agents}
        self.on_event = on_event

    def _emit(self, state: Dict[str, Any], agent: str, level: str, msg: str, kpi: Optional[Dict[str, Any]] = None) -> None:
        merged = {"ticket_id": (state.get("ticket") or {}).get("id")}
        if kpi:
            merged.update(kpi)
        ev = make_event(agent, level, msg, merged)
        log = state.get("event_log")
        if isinstance(log, list):
            log.append(ev)
        if self.on_event:
            self.on_event(ev)

    def run(self, steps: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        for step in steps:
            agent_name = step["agent"]
            key = step.get("key", agent_name)
            when = step.get("when")

            if when is not None:
                try:
                    if not when(state):
                        continue
                except Exception:
                    continue

            agent = self.agents.get(agent_name)
            if agent is None:
                self._emit(state, "Orchestrator", "crit", f"Missing agent: {agent_name}")
                continue

            self._emit(state, agent.name, "info", f"Started: {agent.role}")
            context = {"state": state, "crew": self}
            result = agent.run(
                task=step.get("task", agent.role),
                context=context,
                api_key=state.get("api_key"),
            )
            state[key] = result

            level = "crit" if isinstance(result, dict) and result.get("error") else "ok"
            msg = f"Finished: {agent.role}"
            if isinstance(result, dict) and result.get("error"):
                msg = f"Failed: {agent.role} ({result.get('error')})"
            self._emit(state, agent.name, level, msg)

        return state


# -----------------------------------------------------------------------------
# Agent deterministic demo functions
# -----------------------------------------------------------------------------

def intake_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    db = state.get("db")
    dup = 0
    if db is not None:
        try:
            dup = find_duplicate(db, ticket)
        except Exception:
            dup = 0
    req_id = bool(quick_requires_identity(ticket))
    emit_event(state, "IntakePlanner", "info", f"Planned intake for ticket {ticket.get('id')}.", {"ticket_id": ticket.get("id")})
    return {
        "queue": "L1",
        "duplicate_of": int(dup),
        "requires_identity": req_id,
    }


def classifier_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    res = classify_ticket(ticket)
    emit_event(
        state,
        "Classifier",
        "info",
        f"Ticket {ticket.get('id')} classified as {res['category']} confidence {res['confidence']:.2f}.",
        {"ticket_id": ticket.get("id"), "confidence": res["confidence"]},
    )
    return res


def retriever_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    query = f"{ticket.get('subject', '')} {ticket.get('body', '')}"
    rb = retrieve_runbook(query)
    emit_event(
        state,
        "KnowledgeRetriever",
        "ok",
        f"Selected runbook '{rb['title']}' score {rb['score']:.2f}.",
        {"ticket_id": ticket.get("id"), "runbook": rb["slug"]},
    )
    return {
        "slug": rb["slug"],
        "title": rb["title"],
        "score": float(rb["score"]),
        "category": rb["category"],
        "markdown": rb["markdown"],
    }


def verifier_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    user = state.get("user", {}) or {}
    ticket = state.get("ticket", {}) or {}
    auto_demo = bool(state.get("auto_demo", True))
    answers = state.get("answers", None)

    labels: List[Tuple[str, str]] = []
    if user.get("q1_label"):
        labels.append((user["q1_label"], user["q1_answer_hash"]))
    if user.get("q2_label"):
        labels.append((user["q2_label"], user["q2_answer_hash"]))

    if not labels:
        emit_event(state, "IdentityVerifier", "crit", "No mock directory challenges available.", {"ticket_id": ticket.get("id")})
        return {"status": "failed", "challenges": []}

    if (not auto_demo) and answers is None:
        challenges = [{"label": lab, "correct": None, "answer_masked": ""} for lab, _ in labels]
        emit_event(state, "IdentityVerifier", "warn", "Issued two challenge questions to the requester.", {"ticket_id": ticket.get("id")})
        return {"status": "pending", "challenges": challenges}

    passed = True
    challenges = []
    for lab, h in labels:
        if auto_demo:
            ans = demo_answer_for_label(user, lab)
            if ticket.get("demo_flag") == "wrong_verification":
                ans = f"wrong-demo-{lab}"
        else:
            ans = (answers or {}).get(lab, "")

        ok = hash_answer(lab, ans) == h
        passed = passed and ok
        challenges.append({"label": lab, "correct": bool(ok), "answer_masked": mask_answer(ans)})

    status = "passed" if passed else "failed"
    level = "ok" if passed else "crit"
    emit_event(state, "IdentityVerifier", level, f"Identity verification {status}.", {"ticket_id": ticket.get("id"), "verification": status})
    return {"status": status, "challenges": challenges}


def critic_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    intake = state.get("intake", {}) or {}
    cls = state.get("classifier", {}) or {}
    rb = state.get("retriever", {}) or {}
    ver = state.get("verifier", {}) or {}

    reasons: List[str] = []
    evidence: List[str] = []

    duplicate_of = int(intake.get("duplicate_of", 0) or 0)
    confidence = float(cls.get("confidence", 0.0) or 0.0)
    category = str(cls.get("category", "other"))
    score = float(rb.get("score", 0.0) or 0.0)
    ver_status = str(ver.get("status", ""))

    evidence.append(f"Classifier confidence={confidence:.2f}")
    evidence.append(f"Runbook match score={score:.2f}")
    evidence.append(f"Identity verification={ver_status or 'not required'}")

    if duplicate_of > 0:
        decision = "duplicate"
        reasons.append(f"Duplicate of ticket {duplicate_of}.")
    elif confidence < 0.70:
        decision = "escalate"
        reasons.append("Confidence is below the 0.70 resolution threshold.")
    elif category in HIGH_RISK_CATEGORIES and ver_status == "pending":
        decision = "await_verification"
        reasons.append("High-risk category requires identity challenge answers.")
    elif category in HIGH_RISK_CATEGORIES and ver_status != "passed":
        decision = "escalate"
        reasons.append("High-risk category did not pass identity verification.")
    elif score < 0.02:
        decision = "escalate"
        reasons.append("No strong runbook evidence was found.")
    else:
        decision = "execute"
        reasons.append("Deterministic evidence supports a safe lab resolution.")

    emit_event(state, "Critic", "warn" if decision != "execute" else "ok", f"Critic decision: {decision}.", {"ticket_id": (state.get('ticket') or {}).get('id'), "decision": decision})
    return {
        "decision": decision,
        "reasons": reasons,
        "evidence": evidence,
    }


def mask_action(action: Dict[str, Any]) -> Dict[str, Any]:
    masked = dict(action)
    if "payload" in masked and isinstance(masked["payload"], dict):
        masked["payload"] = {k: mask_secret(v) for k, v in masked["payload"].items()}
    return masked


def build_action(state: Dict[str, Any]) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    user = state.get("user", {}) or {}
    cls = state.get("classifier", {}) or {}
    rb = state.get("retriever", {}) or {}
    intake = state.get("intake", {}) or {}

    category = str(cls.get("category", "other"))
    if int(intake.get("duplicate_of", 0) or 0) > 0:
        action_type = "close_duplicate"
        risk = "low"
        payload = {"duplicate_of": int(intake.get("duplicate_of", 0))}
    else:
        action_type = ACTION_MAP.get(category, "provide_guidance")
        risk = "high" if category in HIGH_RISK_CATEGORIES else ("medium" if action_type in USER_CHANGING_ACTIONS else "low")
        payload = {}

        if action_type == "reset_password":
            token = f"OTP-{sha256(f'otp|{ticket.get('id')}|{user.get('email')}')[:10].upper()}"
            payload = {"one_time_token": token}
        elif action_type == "unlock_account":
            payload = {"unlock": True}
        elif action_type == "reset_mfa":
            payload = {"reenrollment_marker": f"REENROLL-{sha256(f'mfa|{ticket.get('id')}')[:8].upper()}"}
        elif action_type == "push_vpn_profile":
            payload = {"profile": "lab-profile-v2"}
        elif action_type == "push_wifi_profile":
            payload = {"profile": "lab-wifi-v2"}
        elif action_type == "set_default_printer":
            payload = {"printer": user.get("printer_queue") or "PRN-LAB-DEFAULT"}
        elif action_type == "create_software_request":
            payload = {"request_id": f"REQ-SW-{ticket.get('id')}"}
        elif action_type == "recommend_disk_cleanup":
            payload = {"plan": "temp-files-recycle-bin-lab-storage"}
        elif action_type == "reset_mail_profile":
            payload = {"profile_action": "recreate-local-mail-profile-guidance"}
        elif action_type == "grant_lab_access":
            payload = {"profile": "lab-read"}

    consequential = (risk in {"high", "medium"}) or (action_type in USER_CHANGING_ACTIONS)

    return {
        "type": action_type,
        "ticket_id": ticket.get("id"),
        "target": user.get("email") or ticket.get("user_email"),
        "category": category,
        "runbook": rb.get("slug"),
        "risk": risk,
        "consequential": bool(consequential),
        "payload": payload,
    }


def apply_resolution(db: sqlite3.Connection, state: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    user = state.get("user", {}) or {}
    cls = state.get("classifier", {}) or {}
    rb = state.get("retriever", {}) or {}
    intake = state.get("intake", {}) or {}
    ver = state.get("verifier", {}) or {}

    action = gate["action"]
    status = gate["status"]
    ticket_id = int(ticket.get("id", 0))
    action_type = action.get("type", "provide_guidance")
    duplicate_of = int(intake.get("duplicate_of", 0) or 0)

    base_summary = (
        f"Simulated action '{action_type}' for {user.get('email', ticket.get('user_email'))} "
        f"using runbook '{rb.get('title', 'N/A')}'."
    )

    if status == "pending_approval":
        summary = base_summary + " Waiting for explicit human approval. No lab directory change was written."
        update_ticket(
            db,
            ticket_id,
            category=cls.get("category"),
            urgency=cls.get("urgency"),
            sentiment=cls.get("sentiment"),
            confidence=cls.get("confidence"),
            runbook_slug=rb.get("slug"),
            runbook_title=rb.get("title"),
            duplicate_of=duplicate_of,
            verification_status=ver.get("status"),
            action_status="pending_approval",
            status="Awaiting approval",
            resolution_summary=summary,
            processed=1,
        )
        append_audit(
            db,
            actor="system:approval-gate",
            action="approval_gate",
            target=str(user.get("email") or ticket.get("user_email")),
            status="pending_approval",
            details={"ticket_id": ticket_id, "action": mask_action(action), "dry_run": gate.get("dry_run")},
        )
        emit_event(state, "Executor", "warn", f"Ticket {ticket_id} pending human approval.", {"ticket_id": ticket_id, "action_status": "pending_approval"})
        return {"status": "pending_approval", "summary": summary}

    if status == "dry_run":
        if duplicate_of > 0:
            new_status = "Duplicate (dry-run)"
            summary = base_summary + f" Linked to duplicate of ticket {duplicate_of}. Dry-run only; no mock directory field was changed."
        else:
            new_status = "Resolved (dry-run)"
            summary = base_summary + " Dry-run only; no mock directory field was changed."

        update_ticket(
            db,
            ticket_id,
            category=cls.get("category"),
            urgency=cls.get("urgency"),
            sentiment=cls.get("sentiment"),
            confidence=cls.get("confidence"),
            runbook_slug=rb.get("slug"),
            runbook_title=rb.get("title"),
            duplicate_of=duplicate_of,
            verification_status=ver.get("status"),
            action_status="dry_run",
            status=new_status,
            resolution_summary=summary,
            processed=1,
            escalated=0,
        )
        append_audit(
            db,
            actor="system:dry-run",
            action=action_type,
            target=str(user.get("email") or ticket.get("user_email")),
            status="dry_run",
            details={
                "ticket_id": ticket_id,
                "action": mask_action(action),
                "directory_changed": False,
                "classification": cls,
                "runbook": rb.get("slug"),
                "verification_status": ver.get("status"),
            },
        )
        emit_event(state, "Executor", "ok", f"Ticket {ticket_id} resolved in dry-run.", {"ticket_id": ticket_id, "action_status": "dry_run"})
        return {"status": "dry_run", "summary": summary}

    # executed: local lab SQLite only
    try:
        if action_type == "reset_password":
            db.execute("UPDATE users SET password_token = ? WHERE email = ?", (action["payload"].get("one_time_token"), user.get("email")))
        elif action_type == "unlock_account":
            db.execute("UPDATE users SET account_locked = 0 WHERE email = ?", (user.get("email"),))
        elif action_type == "reset_mfa":
            db.execute("UPDATE users SET mfa_secret_masked = ? WHERE email = ?", (action["payload"].get("reenrollment_marker"), user.get("email")))
        elif action_type == "push_vpn_profile":
            db.execute("UPDATE users SET vpn_profile = ? WHERE email = ?", (action["payload"].get("profile"), user.get("email")))
        elif action_type == "push_wifi_profile":
            db.execute("UPDATE users SET wifi_profile = ? WHERE email = ?", (action["payload"].get("profile"), user.get("email")))
        elif action_type == "set_default_printer":
            db.execute("UPDATE users SET printer_queue = ? WHERE email = ?", (action["payload"].get("printer"), user.get("email")))
        elif action_type == "grant_lab_access":
            db.execute("UPDATE users SET access_profile = ? WHERE email = ?", (action["payload"].get("profile"), user.get("email")))
        db.commit()
        directory_changed = action_type in USER_CHANGING_ACTIONS
    except Exception as e:
        directory_changed = False
        base_summary += f" Local lab update skipped: {e}."

    if duplicate_of > 0:
        new_status = "Duplicate"
        summary = base_summary + f" Linked to duplicate of ticket {duplicate_of}. Executed in local lab SQLite only."
    else:
        new_status = "Resolved"
        summary = base_summary + " Executed in local lab SQLite only. No real system was contacted."

    update_ticket(
        db,
        ticket_id,
        category=cls.get("category"),
        urgency=cls.get("urgency"),
        sentiment=cls.get("sentiment"),
        confidence=cls.get("confidence"),
        runbook_slug=rb.get("slug"),
        runbook_title=rb.get("title"),
        duplicate_of=duplicate_of,
        verification_status=ver.get("status"),
        action_status="executed",
        status=new_status,
        resolution_summary=summary,
        processed=1,
        escalated=0,
    )
    append_audit(
        db,
        actor=str(state.get("approver", "local-human")),
        action=action_type,
        target=str(user.get("email") or ticket.get("user_email")),
        status="executed",
        details={
            "ticket_id": ticket_id,
            "action": mask_action(action),
            "directory_changed": bool(directory_changed),
            "approval_record": True,
            "classification": cls,
            "runbook": rb.get("slug"),
            "verification_status": ver.get("status"),
        },
    )
    emit_event(state, "Executor", "ok", f"Ticket {ticket_id} executed in local lab only.", {"ticket_id": ticket_id, "action_status": "executed"})
    return {"status": "executed", "summary": summary}


def executor_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    db = state.get("db")
    critic = state.get("critic", {}) or {}
    decision = critic.get("decision", "escalate")

    if decision not in {"execute", "duplicate"}:
        return {"status": "skipped", "action": {}, "summary": f"Executor skipped because critic decision is {decision}."}

    action = build_action(state)
    gate = approval_gate(
        action,
        approved=bool(state.get("approved", False)),
        dry_run=bool(state.get("dry_run", True)),
    )
    result = apply_resolution(db, state, gate)
    return {
        "status": result["status"],
        "action": action,
        "summary": result["summary"],
    }


def apply_escalation(db: sqlite3.Connection, state: Dict[str, Any], reason: str, summary: str) -> Dict[str, Any]:
    ticket = state.get("ticket", {}) or {}
    cls = state.get("classifier", {}) or {}
    rb = state.get("retriever", {}) or {}
    ver = state.get("verifier", {}) or {}
    ticket_id = int(ticket.get("id", 0))

    update_ticket(
        db,
        ticket_id,
        category=cls.get("category"),
        urgency=cls.get("urgency"),
        sentiment=cls.get("sentiment"),
        confidence=cls.get("confidence"),
        runbook_slug=rb.get("slug"),
        runbook_title=rb.get("title"),
        verification_status=ver.get("status"),
        action_status="escalated",
        status="Escalated",
        escalated=1,
        escalation_reason=reason,
        resolution_summary=summary,
        processed=1,
    )
    append_audit(
        db,
        actor="EscalationAgent",
        action="escalate",
        target=str(ticket.get("user_email")),
        status="escalated",
        details={
            "ticket_id": ticket_id,
            "reason": reason,
            "classification": cls,
            "runbook": rb.get("slug"),
            "verification_status": ver.get("status"),
            "summary_length": len(summary),
        },
    )
    emit_event(state, "EscalationAgent", "crit", f"Ticket {ticket_id} escalated to human analyst.", {"ticket_id": ticket_id, "escalated": 1})
    return {"status": "escalated", "summary": summary, "reason": reason}


def escalation_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    db = state.get("db")
    ticket = state.get("ticket", {}) or {}
    cls = state.get("classifier", {}) or {}
    rb = state.get("retriever", {}) or {}
    ver = state.get("verifier", {}) or {}
    critic = state.get("critic", {}) or {}

    reasons = critic.get("reasons", ["Critic requested human review."])
    reason = "; ".join(str(x) for x in reasons)
    summary = (
        f"Human hand-off for ticket {ticket.get('id')}. "
        f"Predicted category: {cls.get('category')} (confidence {float(cls.get('confidence', 0)):.2f}). "
        f"Urgency: {cls.get('urgency')}. Sentiment: {cls.get('sentiment')}. "
        f"Best runbook: {rb.get('title')} score {float(rb.get('score', 0)):.2f}. "
        f"Identity verification: {ver.get('status', 'not required')}. "
        f"Evidence: {' | '.join(critic.get('evidence', []))}. "
        f"Recommended next step: analyst reviews evidence, confirms ownership if a real host or identity is involved, "
        f"then performs the action manually in the authorized system."
    )
    return apply_escalation(db, state, reason, summary)


def scriptwriter_demo(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    scenario = state.get("scenario", "Normal day")
    db = state.get("db")
    metrics = state.get("metrics", {}) or {}
    events = generate_scripted_events(scenario, db, metrics)
    return {"events": events}


def fallback_sim_events() -> List[Dict[str, Any]]:
    base = datetime.now(timezone.utc)
    rows = [
        ("IntakePlanner", "info", "Queue connected. Synthetic tickets loaded from local lab."),
        ("Classifier", "info", "Password reset ticket classified with high confidence."),
        ("KnowledgeRetriever", "ok", "Runbook Password Reset selected."),
        ("IdentityVerifier", "ok", "Two challenge questions passed in demo mode."),
        ("Critic", "ok", "Evidence supports safe dry-run resolution."),
        ("Executor", "ok", "One-time token generated in local lab only."),
        ("Classifier", "info", "VPN ticket classified."),
        ("KnowledgeRetriever", "ok", "VPN Profile Install selected."),
        ("Executor", "ok", "Simulated VPN profile pushed to lab record."),
        ("Classifier", "info", "Printer ticket classified."),
        ("Executor", "ok", "Default printer set in mock directory."),
        ("Classifier", "warn", "Ambiguous ticket confidence below threshold."),
        ("Critic", "warn", "Skeptic found insufficient evidence."),
        ("EscalationAgent", "crit", "Escalated with complete human hand-off summary."),
        ("Classifier", "info", "MFA reset ticket classified."),
        ("IdentityVerifier", "crit", "Demo wrong-answer path triggered verification failure."),
        ("EscalationAgent", "crit", "Second escalation created for analyst review."),
        ("Orchestrator", "ok", "Auto-resolution KPI updated from real event stream."),
    ]
    out = []
    for i, (agent, level, msg) in enumerate(rows):
        out.append(
            {
                "ts": (base + timedelta(seconds=i)).isoformat(timespec="seconds"),
                "agent": agent,
                "level": level,
                "msg": msg,
                "kpi": {"ticket_id": 1000 + i},
            }
        )
    return out


def generate_scripted_events(scenario: str, db: Optional[sqlite3.Connection], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    base = datetime.now(timezone.utc)

    def add(agent: str, level: str, msg: str, kpi: Optional[Dict[str, Any]] = None) -> None:
        events.append(
            {
                "ts": (base + timedelta(seconds=len(events))).isoformat(timespec="seconds"),
                "agent": agent,
                "level": level,
                "msg": msg,
                "kpi": kpi or {},
            }
        )

    rows: List[Dict[str, Any]] = []
    if db is not None:
        try:
            rows = fetch_all(db, "SELECT * FROM tickets ORDER BY id")
        except Exception:
            rows = []

    if scenario == "Normal day":
        add("IntakePlanner", "info", "Normal day scenario: tickets drop into the L1 queue.")
        selected = rows[:18] if rows else []
        if not selected:
            return fallback_sim_events()
        for r in selected:
            tid = r.get("id")
            cat = r.get("category") or r.get("true_category") or "unknown"
            status = str(r.get("status") or "Open")
            conf = float(r.get("confidence") or 0.0)
            add("Classifier", "info", f"Ticket {tid} classified as {cat} confidence {conf:.2f}.", {"ticket_id": tid})
            if status == "Escalated":
                add("Critic", "warn", f"Ticket {tid} failed skeptic review.", {"ticket_id": tid})
                add("EscalationAgent", "crit", f"Ticket {tid} escalated with summary card.", {"ticket_id": tid})
            else:
                add("KnowledgeRetriever", "ok", f"Ticket {tid} matched runbook {r.get('runbook_title') or 'lab runbook'}.", {"ticket_id": tid})
                add("Executor", "ok", f"Ticket {tid} flipped to {status}.", {"ticket_id": tid})
    elif scenario == "Active attack":
        add("IntakePlanner", "warn", "Active attack scenario: high-risk identity and access tickets surge.")
        high_rows = [r for r in rows if r.get("true_category") in HIGH_RISK_CATEGORIES][:18]
        if not high_rows:
            high_rows = rows[:18]
        if not high_rows:
            return fallback_sim_events()
        for i, r in enumerate(high_rows):
            tid = r.get("id")
            add("Classifier", "warn", f"High-risk ticket {tid} marked for verification.", {"ticket_id": tid})
            if i % 3 == 0:
                add("IdentityVerifier", "crit", f"Ticket {tid} verification challenge failed in simulation.", {"ticket_id": tid})
                add("EscalationAgent", "crit", f"Ticket {tid} escalated to human analyst.", {"ticket_id": tid})
            else:
                add("IdentityVerifier", "ok", f"Ticket {tid} passed simulated challenges.", {"ticket_id": tid})
                add("Executor", "ok", f"Ticket {tid} dry-run resolved in lab only.", {"ticket_id": tid})
    else:  # Recovery
        add("IntakePlanner", "info", "Recovery scenario: backlog clears after controlled lab remediation.")
        selected = rows[-18:] if rows else []
        if not selected:
            return fallback_sim_events()
        for r in selected:
            tid = r.get("id")
            status = str(r.get("status") or "Resolved (dry-run)")
            add("Classifier", "info", f"Recovery ticket {tid} rechecked.", {"ticket_id": tid})
            add("Executor", "ok", f"Ticket {tid} recovered as {status}.", {"ticket_id": tid})
        add("Orchestrator", "ok", "Auto-resolution rate counted up from real event stream.", {"auto_resolution_rate": metrics.get("auto_resolution_rate", 0)})

    if len(events) < 15:
        events.extend(fallback_sim_events()[: 15 - len(events)])
    if len(events) > 25:
        events = events[:25]
    return events


# Build agents and crew
AGENT_LIST = [
    Agent(
        "IntakePlanner",
        "Intake and planning",
        "Plan L1 helpdesk intake. Never assume real-system authority. Prefer dry-run and evidence.",
        intake_demo,
        {"queue": str, "duplicate_of": int, "requires_identity": bool},
    ),
    Agent(
        "Classifier",
        "Ticket classifier",
        "Classify category, urgency, sentiment, and confidence. Return JSON only.",
        classifier_demo,
        {"category": str, "urgency": str, "sentiment": str, "confidence": float},
    ),
    Agent(
        "KnowledgeRetriever",
        "Runbook retriever",
        "Find the best runbook using TF-IDF evidence. Return JSON only.",
        retriever_demo,
        {"slug": str, "title": str, "score": float, "category": str, "markdown": str},
    ),
    Agent(
        "IdentityVerifier",
        "Identity verifier",
        "Before identity-sensitive actions, require two challenge questions from the mock directory.",
        verifier_demo,
        {"status": str, "challenges": list},
    ),
    Agent(
        "Critic",
        "Skeptic critic",
        "Try to disprove the proposed resolution. Escalate when evidence is weak or safety rules fail.",
        critic_demo,
        {"decision": str, "reasons": list, "evidence": list},
    ),
    Agent(
        "Executor",
        "Safe executor",
        "Act only through approval_gate. Dry-run by default. No real OS or network commands.",
        executor_demo,
        {"status": str, "action": dict, "summary": str},
    ),
    Agent(
        "EscalationAgent",
        "Human hand-off",
        "Produce a complete analyst summary with evidence when confidence is low or risk is high.",
        escalation_demo,
        {"status": str, "summary": str, "reason": str},
    ),
    Agent(
        "Scriptwriter",
        "Simulation scriptwriter",
        "Generate 15-25 JSON events for the Simulation Studio when real events are unavailable.",
        scriptwriter_demo,
        {"events": list},
    ),
]

CREW = Crew(AGENT_LIST)
SCRIPTWRITER = next(a for a in AGENT_LIST if a.name == "Scriptwriter")

STEPS = [
    {"agent": "IntakePlanner", "key": "intake"},
    {"agent": "Classifier", "key": "classifier"},
    {"agent": "KnowledgeRetriever", "key": "retriever"},
    {
        "agent": "IdentityVerifier",
        "key": "verifier",
        "when": lambda s: (
            not int((s.get("intake") or {}).get("duplicate_of", 0) or 0)
            and (
                bool((s.get("intake") or {}).get("requires_identity", False))
                or str((s.get("classifier") or {}).get("category", "")) in HIGH_RISK_CATEGORIES
            )
        ),
    },
    {"agent": "Critic", "key": "critic"},
    {
        "agent": "Executor",
        "key": "executor",
        "when": lambda s: str((s.get("critic") or {}).get("decision", "")) in {"execute", "duplicate"},
    },
    {
        "agent": "EscalationAgent",
        "key": "escalation",
        "when": lambda s: str((s.get("critic") or {}).get("decision", "")) == "escalate",
    },
]


def process_ticket(
    db: sqlite3.Connection,
    ticket: Dict[str, Any],
    dry_run: bool = True,
    auto_demo: bool = True,
    api_key: Optional[str] = None,
    event_log: Optional[List[Dict[str, Any]]] = None,
    answers: Optional[Dict[str, str]] = None,
    approved: bool = False,
    approver: str = "local-human",
) -> Dict[str, Any]:
    if event_log is None:
        event_log = []

    user = fetch_one(db, "SELECT * FROM users WHERE email = ?", (ticket.get("user_email"),))
    state: Dict[str, Any] = {
        "ticket": ticket,
        "user": user,
        "db": db,
        "dry_run": bool(dry_run),
        "auto_demo": bool(auto_demo),
        "answers": answers,
        "approved": bool(approved),
        "approver": approver,
        "api_key": api_key,
        "event_log": event_log,
    }

    CREW.run(STEPS, state)

    critic = state.get("critic", {}) or {}
    decision = str(critic.get("decision", ""))

    if not decision or critic.get("error"):
        fallback_escalate(db, state, "Agent pipeline failed or returned no decision.")
        state["critic"] = {"decision": "escalate", "reasons": ["Pipeline failure."], "evidence": []}
        return state

    if decision == "await_verification":
        ticket_id = int(ticket.get("id", 0))
        update_ticket(
            db,
            ticket_id,
            category=(state.get("classifier") or {}).get("category"),
            urgency=(state.get("classifier") or {}).get("urgency"),
            sentiment=(state.get("classifier") or {}).get("sentiment"),
            confidence=(state.get("classifier") or {}).get("confidence"),
            runbook_slug=(state.get("retriever") or {}).get("slug"),
            runbook_title=(state.get("retriever") or {}).get("title"),
            verification_status="pending",
            action_status="awaiting_verification",
            status="Awaiting verification",
            resolution_summary="Two challenge questions were issued. No sensitive action was performed.",
            processed=1,
        )
        append_audit(
            db,
            actor="IdentityVerifier",
            action="issue_challenges",
            target=str(ticket.get("user_email")),
            status="pending",
            details={"ticket_id": ticket_id, "challenges": [c.get("label") for c in (state.get("verifier") or {}).get("challenges", [])]},
        )
        emit_event(state, "Orchestrator", "warn", f"Ticket {ticket_id} awaiting identity verification.", {"ticket_id": ticket_id})

    return state


def fallback_escalate(db: sqlite3.Connection, state: Dict[str, Any], reason: str) -> None:
    ticket = state.get("ticket", {}) or {}
    summary = (
        f"Fallback escalation for ticket {ticket.get('id')}. Reason: {reason}. "
        f"A human analyst should review the raw ticket, mock directory record, and audit trail."
    )
    apply_escalation(db, state, reason, summary)


def run_batch(
    db: sqlite3.Connection,
    tickets: List[Dict[str, Any]],
    dry_run: bool,
    auto_demo: bool,
    event_log: List[Dict[str, Any]],
) -> None:
    for t in tickets:
        try:
            process_ticket(
                db=db,
                ticket=t,
                dry_run=dry_run,
                auto_demo=auto_demo,
                event_log=event_log,
            )
        except Exception as e:
            tid = int(t.get("id", 0))
            event_log.append(make_event("Orchestrator", "crit", f"Ticket {tid} failed safely: {e}", {"ticket_id": tid}))
            try:
                update_ticket(
                    db,
                    tid,
                    status="Escalated",
                    escalated=1,
                    action_status="error",
                    escalation_reason="Processing exception",
                    resolution_summary=f"Automated processing failed safely. Human review required. Error: {e}",
                    processed=1,
                )
                append_audit(
                    db,
                    actor="Orchestrator",
                    action="processing_error",
                    target=str(t.get("user_email")),
                    status="escalated",
                    details={"ticket_id": tid, "error": str(e)},
                )
            except Exception:
                pass


def replay_events(db: sqlite3.Connection, event_log: List[Dict[str, Any]]) -> None:
    rows = fetch_all(db, "SELECT * FROM tickets ORDER BY id LIMIT 20")
    base = datetime.now(timezone.utc)
    for i, r in enumerate(rows):
        event_log.append(
            {
                "ts": (base + timedelta(seconds=i)).isoformat(timespec="seconds"),
                "agent": "Classifier",
                "level": "info",
                "msg": f"Replayed ticket {r['id']} as {r.get('category') or r.get('true_category')}.",
                "kpi": {"ticket_id": r["id"]},
            }
        )
        status = str(r.get("status") or "Open")
        agent = "EscalationAgent" if status == "Escalated" else "Executor"
        level = "crit" if status == "Escalated" else "ok"
        event_log.append(
            {
                "ts": (base + timedelta(seconds=i + 0.5)).isoformat(timespec="seconds"),
                "agent": agent,
                "level": level,
                "msg": f"Replayed outcome: {status}.",
                "kpi": {"ticket_id": r["id"]},
            }
        )


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(db: sqlite3.Connection) -> Dict[str, Any]:
    df = pd.read_sql_query("SELECT * FROM tickets", db)
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "auto_resolution_rate": 0.0,
            "classification_accuracy": 0.0,
            "wrong_resolution_rate": 0.0,
            "escalation_quality": 100.0,
            "time_saved_minutes": 0,
            "csat_estimate": 3.0,
            "escalated_count": 0,
            "resolved_count": 0,
        }

    status = df["status"].fillna("").astype(str)
    resolved_mask = status.str.startswith(("Resolved", "Duplicate"))
    escalated_mask = status.eq("Escalated")
    awaiting_mask = status.str.startswith("Awaiting")

    resolved_count = int(resolved_mask.sum())
    escalated_count = int(escalated_mask.sum())
    awaiting_count = int(awaiting_mask.sum())

    auto_rate = resolved_count / total * 100.0

    pred = df["category"].fillna("").astype(str)
    true = df["true_category"].fillna("").astype(str)
    accuracy = float((pred == true).mean() * 100.0)

    high_true = true.isin(HIGH_RISK_CATEGORIES)
    conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    ver_status = df["verification_status"].fillna("").astype(str)

    wrong_mask = (
        resolved_mask
        & (~status.str.startswith("Duplicate"))
        & (
            (conf < 0.70)
            | (high_true & (ver_status != "passed"))
            | (pred != true)
        )
    )
    wrong_rate = float(wrong_mask.sum() / total * 100.0)

    summaries = df["resolution_summary"].fillna("").astype(str)
    if escalated_count > 0:
        esc_quality = float(((escalated_mask) & (summaries.str.len() >= 80)).sum() / escalated_count * 100.0)
    else:
        esc_quality = 100.0

    time_saved = int(resolved_count * 11 + escalated_count * 2 + awaiting_count * 1)

    esc_rate = escalated_count / total
    pos_rate = float((df["sentiment"].fillna("") == "positive").mean())
    csat = 3.6 + 1.3 * (auto_rate / 100.0) - 1.0 * esc_rate + 0.2 * pos_rate
    csat = max(1.0, min(5.0, csat))

    return {
        "total": total,
        "auto_resolution_rate": round(auto_rate, 2),
        "classification_accuracy": round(accuracy, 2),
        "wrong_resolution_rate": round(wrong_rate, 2),
        "escalation_quality": round(esc_quality, 2),
        "time_saved_minutes": time_saved,
        "csat_estimate": round(csat, 2),
        "escalated_count": escalated_count,
        "resolved_count": resolved_count,
    }


# =============================================================================
# Simulation Studio rendering and MP4 export
# =============================================================================

_FONT_CACHE: Dict[Tuple[int, bool], Any] = {}


def get_font(size: int, bold: bool = False) -> Any:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    candidates = [
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]

    font = None
    for path in candidates:
        try:
            if path and os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break
        except Exception:
            continue

    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    _FONT_CACHE[key] = font
    return font


def level_color(level: str) -> Tuple[int, int, int]:
    return {
        "info": (0, 229, 255),
        "ok": (57, 255, 136),
        "warn": (255, 179, 0),
        "crit": (255, 77, 109),
    }.get(str(level), (154, 167, 184))


def draw_panel(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], fill: Tuple[int, int, int]) -> None:
    try:
        draw.rounded_rectangle(xy, radius=12, fill=fill, outline=(32, 42, 58), width=1)
    except AttributeError:
        draw.rectangle(xy, fill=fill, outline=(32, 42, 58), width=1)


def render_frame(
    title: str,
    agents: List[Dict[str, str]],
    events: List[Dict[str, Any]],
    ev_index: int,
    kpi_fn: Any,
    progress: float,
) -> Image.Image:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), (7, 9, 15))
    draw = ImageDraw.Draw(img)

    title_font = get_font(28, bold=True)
    sub_font = get_font(16)
    small_font = get_font(13)
    mono_font = get_font(14)
    kpi_font = get_font(26, bold=True)

    current = events[ev_index] if 0 <= ev_index < len(events) else (events[-1] if events else {})
    active_agent = str(current.get("agent", ""))

    draw.text((24, 22), title, font=title_font, fill=(0, 229, 255))
    pct = int(max(0.0, min(1.0, progress)) * 100)
    draw.text((24, 58), f"Progress {pct}% | local lab | dry-run by default | no real system changes", font=sub_font, fill=(143, 163, 191))

    draw.rectangle((24, 84, 1256, 98), fill=(14, 19, 30), outline=(32, 42, 58))
    bar_w = int((1256 - 24) * max(0.0, min(1.0, progress)))
    draw.rectangle((24, 84, 24 + bar_w, 98), fill=(0, 229, 255))

    # Left roster
    draw_panel(draw, (20, 112, 328, 692), (14, 19, 30))
    draw.text((34, 124), "Agent roster", font=sub_font, fill=(57, 255, 136))
    y = 154
    for a in agents:
        name = str(a.get("name", ""))
        role = str(a.get("role", ""))
        active = name == active_agent
        fill = (0, 58, 70) if active else (10, 14, 22)
        draw_panel(draw, (32, y, 316, y + 54), fill)
        draw.text((42, y + 8), name, font=mono_font, fill=(0, 229, 255) if active else (215, 227, 244))
        draw.text((42, y + 28), role, font=small_font, fill=(143, 163, 191))
        y += 62

    # Center log
    draw_panel(draw, (344, 112, 944, 692), (14, 19, 30))
    draw.text((358, 124), "Live agent log", font=sub_font, fill=(57, 255, 136))
    start = max(0, ev_index - 14)
    visible = events[start:ev_index + 1]
    y = 154
    for ev in visible:
        if y > 660:
            break
        lvl = str(ev.get("level", "info"))
        color = level_color(lvl)
        agent = str(ev.get("agent", "?"))
        msg = str(ev.get("msg", ""))
        head = f"[{lvl.upper()}] {agent}"
        draw.text((358, y), head, font=mono_font, fill=color)
        y += 20
        for line in textwrap.wrap(msg, width=68)[:4]:
            if y > 660:
                break
            draw.text((372, y), line, font=small_font, fill=(215, 227, 244))
            y += 18
        y += 8

    # Right KPIs
    draw_panel(draw, (960, 112, 1260, 692), (14, 19, 30))
    draw.text((974, 124), "KPI cards", font=sub_font, fill=(57, 255, 136))

    try:
        if callable(kpi_fn):
            kpis = kpi_fn(ev_index, max(1, len(events)))
        elif isinstance(kpi_fn, dict):
            kpis = kpi_fn
        else:
            kpis = {}
    except Exception:
        kpis = {}

    y = 154
    for k, v in list(kpis.items())[:4]:
        draw_panel(draw, (972, y, 1248, y + 104), (10, 14, 22))
        draw.text((986, y + 12), str(k), font=small_font, fill=(143, 163, 191))
        if isinstance(v, float):
            val_text = f"{v:,.1f}"
        elif isinstance(v, int):
            val_text = f"{v:,}"
        else:
            val_text = str(v)
        draw.text((986, y + 42), val_text, font=kpi_font, fill=(0, 229, 255))
        y += 122

    draw.text((24, 698), "TicketPilot | synthetic data only | approval-gated executor | hash-chained audit", font=small_font, fill=(92, 108, 130))
    return img


def export_mp4(
    title: str,
    agents: List[Dict[str, str]],
    events: List[Dict[str, Any]],
    kpi_fn: Any,
    fps: int = 10,
    hold: int = 8,
) -> bytes:
    if not IMAGEIO_AVAILABLE or imageio is None:
        raise RuntimeError("imageio/imageio-ffmpeg is not available. Install requirements.txt and restart Streamlit.")
    if not events:
        raise RuntimeError("No simulation events available.")

    tmp = tempfile.NamedTemporaryFile(prefix="ticketpilot_", suffix=".mp4", delete=False)
    tmp.close()
    path = tmp.name

    try:
        writer = imageio.get_writer(path, fps=int(fps), codec="libx264", pixelformat="yuv420p")
        n = max(1, len(events))
        for i in range(n):
            img = render_frame(title, agents, events, i, kpi_fn, (i + 1) / n)
            arr = np.ascontiguousarray(np.array(img, dtype=np.uint8))
            for _ in range(max(1, int(hold))):
                writer.append_data(arr)
        writer.close()

        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def make_progress_kpi_fn(metrics: Dict[str, Any]) -> Callable[[int, int], Dict[str, Any]]:
    total = max(1, int(metrics.get("total", 100)))
    auto = float(metrics.get("auto_resolution_rate", 98.0))
    esc = int(metrics.get("escalated_count", 2))
    saved = float(metrics.get("time_saved_minutes", 1000))

    def fn(i: int, n: int) -> Dict[str, Any]:
        p = (i + 1) / max(1, n)
        return {
            "Auto-resolution %": round(auto * p, 1),
            "Tickets seen": int(total * p),
            "Escalations": int(esc * min(1.0, p * 1.15)),
            "Minutes saved": int(saved * p),
        }

    return fn


SIM_AGENTS = [{"name": a.name, "role": a.role} for a in AGENT_LIST if a.name != "Scriptwriter"]


def get_simulation_events(scenario: str, db: sqlite3.Connection, metrics: Dict[str, Any], max_events: int = 22) -> List[Dict[str, Any]]:
    real = st.session_state.get("events", []) or []

    if scenario == "Normal day" and len(real) >= 15:
        return real[-max_events:]

    if scenario == "Active attack":
        filtered = [
            e for e in real
            if e.get("agent") in {"IdentityVerifier", "Critic", "EscalationAgent"} or e.get("level") in {"warn", "crit"}
        ]
        if len(filtered) >= 15:
            return filtered[-max_events:]

    if scenario == "Recovery":
        filtered = [e for e in real if e.get("agent") == "Executor" and e.get("level") == "ok"]
        if len(filtered) >= 15:
            return filtered[-max_events:]

    ctx = {"state": {"scenario": scenario, "db": db, "metrics": metrics, "event_log": []}}
    res = SCRIPTWRITER.run("Generate simulation events", ctx, api_key=None)
    events = res.get("events", []) if isinstance(res, dict) else []
    if not events:
        events = fallback_sim_events()
    return events[:max_events]


# =============================================================================
# Streamlit UI
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #07090F;
            color: #D7E3F4;
            font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
        }
        h1, h2, h3, h4, h5, p, li, label {
            color: #D7E3F4 !important;
            font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
        }
        div[data-testid="stMetricValue"] {
            color: #00E5FF !important;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }
        div[data-testid="stMetricLabel"] {
            color: #8FA3BF !important;
        }
        [data-testid="stSidebar"] {
            background-color: #0B0F17;
            border-right: 1px solid #1F2A3A;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #0E131E;
            border-radius: 8px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #8FA3BF !important;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }
        .stTabs [aria-selected="true"] {
            color: #39FF88 !important;
        }
        button, .stButton > button, .stFormSubmitButton > button {
            background-color: #0E131E;
            color: #00E5FF;
            border: 1px solid #1F2A3A;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }
        button:hover, .stButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: #39FF88;
            color: #39FF88;
        }
        div[data-testid="stExpander"] details {
            background-color: #0E131E;
            border: 1px solid #1F2A3A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_db() -> sqlite3.Connection:
    if "db" not in st.session_state:
        st.session_state.db = get_db_connection(DB_PATH)
    return st.session_state.db


def render_sidebar(db: sqlite3.Connection) -> Dict[str, Any]:
    st.sidebar.header("TicketPilot controls")
    st.sidebar.caption("Local-first L1 helpdesk automation. Synthetic data only.")

    demo_mode = st.sidebar.checkbox("Demo Mode (deterministic)", value=True)
    live_mode = st.sidebar.checkbox("Try optional Gemini if key exists", value=False)
    api_key_raw = st.sidebar.text_input("Optional Gemini API key", type="password", value="", disabled=not live_mode)
    api_key = api_key_raw.strip() if live_mode and api_key_raw else None

    dry_run = st.sidebar.checkbox("Dry-run for manual actions", value=True)
    auto_demo = st.sidebar.checkbox("Auto-answer demo challenges", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset lab database"):
        reset_db(db)
        st.session_state.events = []
        st.session_state.pending_challenges = {}
        st.session_state.pending_answers = {}
        st.session_state.last_verifier = {}
        st.session_state.demo_video = None
        st.session_state.sim_video = None
        st.session_state.demo_video_error = None
        st.rerun()

    st.sidebar.caption(
        "Safety: all actions are simulated against local SQLite. "
        "No OS commands, no real password changes, no real network blocking. "
        "Consequential actions require explicit human approval when dry-run is off."
    )

    return {
        "demo_mode": demo_mode,
        "live_mode": live_mode,
        "api_key": api_key,
        "dry_run": dry_run,
        "auto_demo": auto_demo,
    }


def ensure_demo(db: sqlite3.Connection, cfg: Dict[str, Any]) -> None:
    if "events" not in st.session_state:
        st.session_state.events = []

    unprocessed = fetch_all(db, "SELECT * FROM tickets WHERE processed = 0 ORDER BY id")
    if unprocessed:
        run_batch(
            db=db,
            tickets=unprocessed,
            dry_run=True,
            auto_demo=True,
            event_log=st.session_state.events,
        )
    elif not st.session_state.events:
        replay_events(db, st.session_state.events)


def ensure_demo_video(db: sqlite3.Connection, metrics: Dict[str, Any]) -> None:
    if "demo_video" in st.session_state:
        return

    try:
        events = get_simulation_events("Normal day", db, metrics, max_events=18)
        kpi_fn = make_progress_kpi_fn(metrics)
        video = export_mp4(
            title="TicketPilot Live Resolution Simulation",
            agents=SIM_AGENTS,
            events=events,
            kpi_fn=kpi_fn,
            fps=10,
            hold=8,
        )
        st.session_state.demo_video = video
        st.session_state.demo_events = events
        st.session_state.demo_video_error = None
    except Exception as e:
        st.session_state.demo_video = None
        st.session_state.demo_video_error = str(e)


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#07090F",
        plot_bgcolor="#0E131E",
        font=dict(family="JetBrains Mono, Consolas, monospace", color="#D7E3F4", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    try:
        fig.update_xaxes(gridcolor="#1F2A3A")
        fig.update_yaxes(gridcolor="#1F2A3A")
    except Exception:
        pass
    return fig


def kpi_row(metrics: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Auto-resolution rate", f"{metrics['auto_resolution_rate']:.1f}%")
    c2.metric("Classification accuracy", f"{metrics['classification_accuracy']:.1f}%")
    c3.metric("Wrong-resolution rate", f"{metrics['wrong_resolution_rate']:.1f}%")
    c4.metric("Escalation quality", f"{metrics['escalation_quality']:.1f}%")
    c5.metric("Estimated time saved", f"{metrics['time_saved_minutes']:,} min")


def simulation_tab(db: sqlite3.Connection, metrics: Dict[str, Any]) -> None:
    st.subheader("Simulation Studio")
    st.caption("This tab uses the app's real event stream when available. Otherwise the Scriptwriter agent generates a deterministic fallback.")

    scenario = st.selectbox("Scenario", ["Normal day", "Active attack", "Recovery"])
    speed = st.slider("Seconds per event", min_value=0.2, max_value=2.0, value=0.45, step=0.05)

    events = get_simulation_events(scenario, db, metrics, max_events=22)
    st.write(f"Events prepared: **{len(events)}**")

    col1, col2, col3 = st.columns(3)

    if col1.button("Run live simulation"):
        if not events:
            st.warning("No events available for this scenario.")
        else:
            placeholder = st.empty()
            progress = st.progress(0.0, text="Starting simulation...")
            n = len(events)
            kpi_fn = make_progress_kpi_fn(metrics)
            for i in range(n):
                img = render_frame(
                    title=f"TicketPilot | {scenario}",
                    agents=SIM_AGENTS,
                    events=events,
                    ev_index=i,
                    kpi_fn=kpi_fn,
                    progress=(i + 1) / n,
                )
                arr = np.array(img, dtype=np.uint8)
                placeholder.image(arr, channels="RGB", use_container_width=True)
                progress.progress((i + 1) / n, text=f"Event {i + 1}/{n}: {events[i].get('agent', '')}")
                time.sleep(speed)
            st.success("Live simulation finished.")

    if col2.button("Record video for this scenario"):
        try:
            with st.spinner("Encoding H.264 MP4 with bundled ffmpeg..."):
                kpi_fn = make_progress_kpi_fn(metrics)
                video = export_mp4(
                    title=f"TicketPilot | {scenario}",
                    agents=SIM_AGENTS,
                    events=events,
                    kpi_fn=kpi_fn,
                    fps=10,
                    hold=8,
                )
            st.session_state.sim_video = video
            st.session_state.sim_events = events
            st.rerun()
        except Exception as e:
            st.error(f"Video recording failed safely: {e}")

    if col3.button("Clear recorded video"):
        st.session_state.sim_video = None
        st.session_state.sim_events = None
        st.rerun()

    video = st.session_state.get("sim_video") or st.session_state.get("demo_video")
    download_events = st.session_state.get("sim_events") or st.session_state.get("demo_events") or events

    if video:
        st.video(video, format="video/mp4", autoplay=True, muted=True, loop=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download simulation.mp4", data=video, file_name="simulation.mp4", mime="video/mp4")
        d2.download_button(
            "Download events.json",
            data=json.dumps(download_events, indent=2, default=str).encode("utf-8"),
            file_name="events.json",
            mime="application/json",
        )
    else:
        st.info("No video is currently stored in this session. Click Record video for this scenario.")

    err = st.session_state.get("demo_video_error")
    if err:
        st.warning(f"Automatic startup video failed safely: {err}")


def ticket_queue_tab(db: sqlite3.Connection) -> None:
    st.subheader("Ticket Queue")
    df = pd.read_sql_query("SELECT * FROM tickets", db)
    if df.empty:
        st.warning("No tickets found. Use Reset lab database in the sidebar.")
        return

    show = df[[
        "id", "created_at", "user_email", "true_category", "category",
        "urgency", "confidence", "status", "verification_status", "action_status"
    ]].copy()
    show["confidence"] = pd.to_numeric(show["confidence"], errors="coerce").round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)

    tid = st.selectbox("Inspect ticket", df["id"].astype(int).tolist())
    t = fetch_one(db, "SELECT * FROM tickets WHERE id = ?", (tid,))
    if not t:
        st.error("Ticket not found.")
        return

    with st.expander("Ticket details", expanded=True):
        st.json(
            {
                "id": t.get("id"),
                "created_at": t.get("created_at"),
                "user_email": t.get("user_email"),
                "subject": t.get("subject"),
                "body": t.get("body"),
                "true_category": t.get("true_category"),
                "predicted_category": t.get("category"),
                "urgency": t.get("urgency"),
                "sentiment": t.get("sentiment"),
                "confidence": t.get("confidence"),
                "status": t.get("status"),
                "verification_status": t.get("verification_status"),
                "action_status": t.get("action_status"),
                "duplicate_of": t.get("duplicate_of"),
                "resolution_summary": t.get("resolution_summary"),
            }
        )

    rb = RUNBOOK_BY_SLUG.get(t.get("runbook_slug") or "")
    if rb:
        with st.expander("Matched runbook", expanded=False):
            st.markdown(rb["markdown"])

    audit = fetch_all(
        db,
        "SELECT id, ts, actor, action, target, status, entry_hash FROM audit WHERE target = ? OR details LIKE ? ORDER BY id DESC LIMIT 12",
        (t.get("user_email"), f"%{tid}%"),
    )
    st.markdown("#### Recent audit records for this ticket")
    st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)


def live_chat_tab(db: sqlite3.Connection, cfg: Dict[str, Any]) -> None:
    st.subheader("Live Resolution Chat")
    df = pd.read_sql_query("SELECT id, subject, status, category FROM tickets", db)
    if df.empty:
        st.warning("No tickets available.")
        return

    tid = st.selectbox("Select ticket", df["id"].astype(int).tolist())
    t = fetch_one(db, "SELECT * FROM tickets WHERE id = ?", (tid,))
    if not t:
        st.error("Ticket not found.")
        return

    st.markdown(f"**Subject:** {t.get('subject')}")
    st.markdown(f"**Status:** {t.get('status')}")
    st.markdown(f"**Body:** {t.get('body')}")

    events = [e for e in st.session_state.get("events", []) if int((e.get("kpi") or {}).get("ticket_id", -1)) == int(tid)]
    if events:
        st.markdown("#### Agent trace")
        for e in events[-25:]:
            color = {"info": "#00E5FF", "ok": "#39FF88", "warn": "#FFB300", "crit": "#FF4D6D"}.get(e.get("level"), "#D7E3F4")
            st.markdown(f"<div style='border-left:3px solid {color}; padding:6px 10px; margin:6px 0; background:#0E131E;'><b>{e.get('ts')} | {e.get('agent')} | {e.get('level')}</b><br>{e.get('msg')}</div>", unsafe_allow_html=True)
    else:
        st.caption("No agent trace stored in this session for this ticket yet.")

    ver = st.session_state.get("last_verifier", {}).get(tid) or st.session_state.get("pending_challenges", {}).get(tid)
    if ver and isinstance(ver, dict) and ver.get("challenges"):
        st.markdown("#### Identity challenges")
        for c in ver.get("challenges", []):
            correct = c.get("correct")
            mark = "passed" if correct is True else ("failed" if correct is False else "pending")
            st.text(f"{c.get('label')} -> {mark} ({c.get('answer_masked') or 'masked'})")

    col1, col2, col3 = st.columns(3)

    if col1.button("Auto-resolve safe demo"):
        state = process_ticket(
            db=db,
            ticket=t,
            dry_run=True,
            auto_demo=True,
            api_key=None,
            event_log=st.session_state.events,
        )
        st.session_state.setdefault("last_verifier", {})[tid] = state.get("verifier", {})
        st.rerun()

    if col2.button("Issue challenges manual"):
        state = process_ticket(
            db=db,
            ticket=t,
            dry_run=bool(cfg["dry_run"]),
            auto_demo=False,
            api_key=cfg.get("api_key"),
            event_log=st.session_state.events,
            answers=None,
        )
        challenges = (state.get("verifier") or {}).get("challenges", [])
        if challenges:
            st.session_state.setdefault("pending_challenges", {})[tid] = challenges
        st.session_state.setdefault("last_verifier", {})[tid] = state.get("verifier", {})
        st.rerun()

    if col3.button("Clear pending"):
        st.session_state.get("pending_challenges", {}).pop(tid, None)
        st.session_state.get("pending_answers", {}).pop(tid, None)
        st.rerun()

    pending = st.session_state.get("pending_challenges", {}).get(tid)
    if pending:
        st.markdown("#### Answer challenges")
        with st.form(f"verify_form_{tid}"):
            inputs = []
            for ch in pending:
                label = ch.get("label", "Challenge")
                val = st.text_input(label, key=f"ans_{tid}_{label}")
                inputs.append((label, val))
            authorize = st.checkbox("Authorize simulated lab action if dry-run is off", value=False)
            submitted = st.form_submit_button("Verify answers and resolve")

            if submitted:
                answers = dict(inputs)
                st.session_state.setdefault("pending_answers", {})[tid] = answers
                state = process_ticket(
                    db=db,
                    ticket=t,
                    dry_run=bool(cfg["dry_run"]),
                    auto_demo=False,
                    api_key=cfg.get("api_key"),
                    event_log=st.session_state.events,
                    answers=answers,
                    approved=bool(authorize),
                )
                st.session_state.setdefault("last_verifier", {})[tid] = state.get("verifier", {})
                if str((state.get("critic") or {}).get("decision", "")) != "await_verification":
                    st.session_state.get("pending_challenges", {}).pop(tid, None)
                st.rerun()

    if str(t.get("status") or "") == "Awaiting approval":
        st.warning("This ticket is waiting for explicit human approval because dry-run is off and the action is consequential.")
        answers = st.session_state.get("pending_answers", {}).get(tid)
        if st.button("Approve and execute lab action"):
            process_ticket(
                db=db,
                ticket=t,
                dry_run=bool(cfg["dry_run"]),
                auto_demo=bool(cfg["auto_demo"]),
                api_key=cfg.get("api_key"),
                event_log=st.session_state.events,
                answers=answers,
                approved=True,
                approver="local-human",
            )
            st.rerun()


def runbook_library_tab() -> None:
    st.subheader("Runbook Library")
    query = st.text_input("Search runbooks with TF-IDF", value="vpn tunnel cannot connect")
    results = retrieve_top_n(query, n=5) if query else []

    if results:
        st.markdown("#### Top matches")
        for r in results:
            st.markdown(f"- **{r['title']}** (`{r['slug']}`) score `{r['score']:.2f}`")

    slugs = [r["slug"] for r in RUNBOOKS]
    default_slug = results[0]["slug"] if results else slugs[0]
    chosen = st.selectbox("Open runbook", slugs, index=slugs.index(default_slug))
    rb = RUNBOOK_BY_SLUG[chosen]
    st.markdown(rb["markdown"])


def metrics_tab(db: sqlite3.Connection, metrics: Dict[str, Any]) -> None:
    st.subheader("Metrics and audit")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tickets", f"{metrics['total']:,}")
    c2.metric("Resolved / duplicate", f"{metrics['resolved_count']:,}")
    c3.metric("Escalated", f"{metrics['escalated_count']:,}")
    c4.metric("CSAT estimate", f"{metrics['csat_estimate']:.2f} / 5.00")

    df = pd.read_sql_query("SELECT * FROM tickets", db)
    if df.empty:
        st.warning("No metrics data.")
        return

    df["outcome"] = df["status"].fillna("").apply(lambda s: "Escalated" if s == "Escalated" else ("Awaiting" if s.startswith("Awaiting") else "Resolved"))

    tab1, tab2, tab3, tab4 = st.tabs(["Categories", "Status", "Confidence", "Audit"])

    with tab1:
        cats = df["true_category"].fillna("unknown").value_counts().reset_index()
        cats.columns = ["Category", "Tickets"]
        fig = px.bar(cats, x="Category", y="Tickets", title="Synthetic ticket categories")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab2:
        status = df["status"].fillna("Unknown").value_counts().reset_index()
        status.columns = ["Status", "Count"]
        fig = px.pie(status, names="Status", values="Count", title="Resolution status distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab3:
        fig = px.histogram(
            df.assign(confidence=pd.to_numeric(df["confidence"], errors="coerce")),
            x="confidence",
            color="outcome",
            nbins=20,
            title="Classifier confidence by outcome",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab4:
        ok, count, last_hash = verify_audit_chain(db)
        if ok:
            st.success(f"Audit chain verified. Records: {count}. Last hash: {last_hash[:16]}...")
        else:
            st.error(f"Audit chain verification failed at record {count}. Last good hash: {last_hash[:16]}...")

        st.download_button(
            "Download audit trail JSON",
            data=audit_to_json(db),
            file_name="ticketpilot_audit.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="TicketPilot", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    db = get_db()

    for key, default in {
        "events": [],
        "pending_challenges": {},
        "pending_answers": {},
        "last_verifier": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    cfg = render_sidebar(db)

    try:
        with st.spinner("Loading synthetic lab data and running deterministic L1 crew..."):
            ensure_demo(db, cfg)
    except Exception as e:
        st.error(f"Demo initialization failed safely: {e}")

    metrics = compute_metrics(db)

    with st.spinner("Rendering live Simulation Studio MP4 from the app event stream..."):
        ensure_demo_video(db, metrics)

    st.title("TicketPilot")
    st.caption("Level-1 helpdesk agent that resolves repetitive tickets in a local synthetic lab. No API key required.")

    kpi_row(metrics)

    tabs = st.tabs(["Simulation Studio", "Ticket Queue", "Live Resolution Chat", "Runbook Library", "Metrics"])
    with tabs[0]:
        simulation_tab(db, metrics)
    with tabs[1]:
        ticket_queue_tab(db)
    with tabs[2]:
        live_chat_tab(db, cfg)
    with tabs[3]:
        runbook_library_tab()
    with tabs[4]:
        metrics_tab(db, metrics)


# =============================================================================
# Embedded pytest tests (run: pytest app.py)
# =============================================================================

def _make_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    seed_users(conn)
    seed_tickets(conn)
    return conn


def test_runbooks_count_is_25() -> None:
    assert len(RUNBOOKS) == 25


def test_classifier_password_reset() -> None:
    res = classify_ticket({"subject": "Password reset", "body": "I forgot my password and need a reset."})
    assert res["category"] == "password_reset"
    assert res["confidence"] >= 0.70


def test_retriever_returns_best_runbook() -> None:
    rb = retrieve_runbook("vpn tunnel cannot connect to lab network")
    assert rb["slug"] in RUNBOOK_BY_SLUG
    assert rb["score"] >= 0.0


def test_approval_gate_states() -> None:
    action = {"type": "reset_password", "consequential": True}
    assert approval_gate(action, approved=False, dry_run=True)["status"] == "dry_run"
    assert approval_gate(action, approved=False, dry_run=False)["status"] == "pending_approval"
    assert approval_gate(action, approved=True, dry_run=False)["status"] == "executed"


def test_audit_hash_chain() -> None:
    db = _make_memory_db()
    append_audit(db, "tester", "seed", "unit", "ok", {"a": 1})
    append_audit(db, "tester", "check", "unit", "ok", {"b": 2})
    ok, count, _ = verify_audit_chain(db)
    assert ok is True
    assert count >= 2


def test_low_confidence_ticket_escalates() -> None:
    db = _make_memory_db()
    row = fetch_one(db, "SELECT * FROM tickets WHERE demo_flag = 'low_confidence' LIMIT 1")
    state = process_ticket(db, row, dry_run=True, auto_demo=True, event_log=[])
    assert state["critic"]["decision"] == "escalate"
    updated = fetch_one(db, "SELECT status FROM tickets WHERE id = ?", (row["id"],))
    assert updated["status"] == "Escalated"


def test_wrong_verification_escalates() -> None:
    db = _make_memory_db()
    row = fetch_one(db, "SELECT * FROM tickets WHERE demo_flag = 'wrong_verification' LIMIT 1")
    state = process_ticket(db, row, dry_run=True, auto_demo=True, event_log=[])
    assert state["verifier"]["status"] == "failed"
    assert state["critic"]["decision"] == "escalate"


def test_duplicate_ticket_closes_without_sensitive_action() -> None:
    db = _make_memory_db()
    row = fetch_one(db, "SELECT * FROM tickets WHERE demo_flag = 'duplicate' LIMIT 1")
    state = process_ticket(db, row, dry_run=True, auto_demo=True, event_log=[])
    assert state["intake"]["duplicate_of"] > 0
    assert state["critic"]["decision"] == "duplicate"
    updated = fetch_one(db, "SELECT status FROM tickets WHERE id = ?", (row["id"],))
    assert str(updated["status"]).startswith("Duplicate")


def test_sensitive_category_requires_verification_when_manual() -> None:
    db = _make_memory_db()
    row = fetch_one(db, "SELECT * FROM tickets WHERE true_category = 'password_reset' AND demo_flag = '' LIMIT 1")

    pending_state = process_ticket(db, row, dry_run=True, auto_demo=False, answers=None, event_log=[])
    assert pending_state["critic"]["decision"] == "await_verification"

    user = fetch_one(db, "SELECT * FROM users WHERE email = ?", (row["user_email"],))
    challenges = pending_state["verifier"]["challenges"]
    answers = {c["label"]: demo_answer_for_label(user, c["label"]) for c in challenges}

    passed_state = process_ticket(db, row, dry_run=True, auto_demo=False, answers=answers, event_log=[])
    assert passed_state["verifier"]["status"] == "passed"
    assert passed_state["critic"]["decision"] == "execute"


if __name__ == "__main__":
    main()