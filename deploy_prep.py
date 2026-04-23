"""
Run this after setup.py + first Gmail auth to print the Railway env vars.

Usage:
    python deploy_prep.py
"""

import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))

errors = []

# Agent config
config = {}
config_path = os.path.join(DIR, "agent_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
else:
    errors.append("agent_config.json not found — run setup.py first")

# Gmail token
token = {}
token_path = os.path.join(DIR, "token.json")
if os.path.exists(token_path):
    with open(token_path) as f:
        token = json.load(f)
else:
    errors.append("token.json not found — run 'python email_agent.py' once locally to authenticate Gmail")

# Gmail client credentials
client_info = {}
creds_path = os.path.join(DIR, "credentials.json")
if os.path.exists(creds_path):
    with open(creds_path) as f:
        raw = json.load(f)
    client_info = raw.get("web") or raw.get("installed") or {}
else:
    errors.append("credentials.json not found — download from Google Cloud Console")

if errors:
    print("⚠️  Missing prerequisites:")
    for e in errors:
        print(f"   • {e}")
    raise SystemExit(1)

print("Copy these into Railway → your service → Variables:\n")
print(f"ANTHROPIC_API_KEY=<your-anthropic-api-key>")
print(f"AGENT_ID={config.get('agent_id', '')}")
print(f"AGENT_VERSION={config.get('agent_version', '')}")
print(f"ENVIRONMENT_ID={config.get('environment_id', '')}")
print(f"GMAIL_CLIENT_ID={client_info.get('client_id', '')}")
print(f"GMAIL_CLIENT_SECRET={client_info.get('client_secret', '')}")
print(f"GMAIL_REFRESH_TOKEN={token.get('refresh_token', '')}")
print()
print("Note: replace <your-anthropic-api-key> with your actual key.")
