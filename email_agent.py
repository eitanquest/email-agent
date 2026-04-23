"""
Email summarizer — run this whenever you want a summary of recent emails.

Prerequisites
-------------
1. pip install anthropic google-api-python-client google-auth-httplib2 google-auth-oauthlib
2. Enable the Gmail API in Google Cloud Console and download credentials.json
   (APIs & Services → Credentials → Create OAuth 2.0 Client ID → Desktop app)
3. Place credentials.json in this directory.
4. Run setup.py once to create the agent and environment.
5. Set ANTHROPIC_API_KEY in your environment.

First run: a browser window will open for Gmail OAuth consent.
Subsequent runs: uses the cached token.json automatically.
"""

import anthropic
import base64
import json
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------

def _gmail_service():
    """Return an authenticated Gmail API service, refreshing/prompting as needed."""
    token_path = os.path.join(DIR, "token.json")
    creds_path = os.path.join(DIR, "credentials.json")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    "credentials.json not found. Download it from Google Cloud Console → "
                    "APIs & Services → Credentials → your OAuth 2.0 Client ID."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _extract_body(payload: dict, max_chars: int = 1500) -> str:
    """Pull plain-text body from a Gmail message payload."""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")[:max_chars]
        # Recurse into nested multipart
        for part in payload["parts"]:
            body = _extract_body(part, max_chars)
            if body:
                return body
    data = payload.get("body", {}).get("data", "")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")[:max_chars]
    return ""


def fetch_emails(max_results: int = 20, query: str = "") -> list[dict]:
    """Fetch recent emails and return structured data."""
    max_results = min(max_results, 50)
    service = _gmail_service()

    list_params: dict = {"userId": "me", "maxResults": max_results}
    if query:
        list_params["q"] = query

    result = service.users().messages().list(**list_params).execute()
    messages = result.get("messages", [])

    emails = []
    for msg in messages:
        full = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()

        headers = {h["name"]: h["value"] for h in full["payload"].get("headers", [])}
        emails.append({
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "snippet": full.get("snippet", ""),
            "body_preview": _extract_body(full["payload"]),
        })

    return emails


# ---------------------------------------------------------------------------
# Agent session
# ---------------------------------------------------------------------------

def run(prompt: str = "Please check my email and give me a concise summary of my most recent emails."):
    config_path = os.path.join(DIR, "agent_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("agent_config.json not found — run setup.py first.")

    with open(config_path) as f:
        config = json.load(f)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    session = client.beta.sessions.create(
        agent={"type": "agent", "id": config["agent_id"], "version": config["agent_version"]},
        environment_id=config["environment_id"],
    )
    print(f"Session: {session.id}\n")

    try:
        # Open stream BEFORE sending the first message (stream-first pattern)
        with client.beta.sessions.stream(session_id=session.id) as stream:
            client.beta.sessions.events.send(
                session_id=session.id,
                events=[{"type": "user.message", "content": [{"type": "text", "text": prompt}]}],
            )

            for event in stream:
                # Agent is writing its summary
                if event.type == "agent.message":
                    for block in event.content:
                        if block.type == "text":
                            print(block.text, end="", flush=True)

                # Agent wants emails — fetch them host-side (credentials never leave this machine)
                elif event.type == "agent.custom_tool_use" and event.tool_name == "get_recent_emails":
                    tool_input = event.input or {}
                    max_results = int(tool_input.get("max_results", 20))
                    query = tool_input.get("query", "")
                    print(f"[Fetching {max_results} emails{' (query: ' + query + ')' if query else ''}...]\n")

                    try:
                        emails = fetch_emails(max_results=max_results, query=query)
                        result_text = json.dumps(emails, indent=2, ensure_ascii=False)
                    except Exception as exc:
                        result_text = f"Error fetching emails: {exc}"

                    client.beta.sessions.events.send(
                        session_id=session.id,
                        events=[
                            {
                                "type": "user.custom_tool_result",
                                "custom_tool_use_id": event.id,
                                "content": [{"type": "text", "text": result_text}],
                            }
                        ],
                    )

                # Session finished normally
                elif event.type == "session.status_idle":
                    if event.stop_reason.type != "requires_action":
                        break

                elif event.type == "session.status_terminated":
                    break

    finally:
        print("\n")
        client.beta.sessions.delete(session_id=session.id)


if __name__ == "__main__":
    run()
