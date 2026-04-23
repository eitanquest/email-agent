"""
Email Summarizer — FastAPI web server with Google OAuth login.

Local dev:
    uvicorn app:app --reload --port 8000

Railway:
    Set env vars listed in README. Railway injects $PORT automatically.
    Start command: uvicorn app:app --host 0.0.0.0 --port $PORT

Required env vars:
    ANTHROPIC_API_KEY   — Anthropic API key
    AGENT_ID            — from setup.py
    AGENT_VERSION       — from setup.py
    ENVIRONMENT_ID      — from setup.py
    GMAIL_CLIENT_ID     — Google OAuth Web Application client ID
    GMAIL_CLIENT_SECRET — Google OAuth Web Application client secret
    SESSION_SECRET      — random secret for signing session cookies
    BASE_URL            — public URL (e.g. https://api-production-f287.up.railway.app)
"""

import asyncio
import base64
import json
import os
import secrets
import threading
from typing import AsyncIterator

import urllib.parse

import anthropic
import requests as http_requests
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from starlette.middleware.sessions import SessionMiddleware

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI()

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, https_only=False)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
REDIRECT_URI = f"{BASE_URL}/auth/callback"

SCOPES = " ".join([
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_agent_config() -> dict:
    agent_id = os.environ.get("AGENT_ID")
    agent_version = os.environ.get("AGENT_VERSION")
    environment_id = os.environ.get("ENVIRONMENT_ID")
    if agent_id and agent_version and environment_id:
        return {"agent_id": agent_id, "agent_version": agent_version, "environment_id": environment_id}
    config_path = os.path.join(DIR, "agent_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("agent_config.json not found and AGENT_ID/AGENT_VERSION/ENVIRONMENT_ID env vars not set.")
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def _creds_from_session(token_dict: dict) -> Credentials:
    return Credentials(
        token=token_dict["token"],
        refresh_token=token_dict.get("refresh_token"),
        token_uri=token_dict.get("token_uri", GOOGLE_TOKEN_URL),
        client_id=token_dict["client_id"],
        client_secret=token_dict["client_secret"],
        scopes=token_dict.get("scopes", SCOPES.split()),
    )


# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------

def _gmail_service(token_dict: dict):
    creds = _creds_from_session(token_dict)
    return build("gmail", "v1", credentials=creds)


def _extract_body(payload: dict, max_chars: int = 1500) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")[:max_chars]
        for part in payload["parts"]:
            body = _extract_body(part, max_chars)
            if body:
                return body
    data = payload.get("body", {}).get("data", "")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")[:max_chars]
    return ""


def fetch_emails(token_dict: dict, max_results: int = 20, query: str = "") -> list[dict]:
    service = _gmail_service(token_dict)
    params: dict = {"userId": "me", "maxResults": min(max_results, 50)}
    if query:
        params["q"] = query

    result = service.users().messages().list(**params).execute()
    messages = result.get("messages", [])

    emails = []
    for msg in messages:
        full = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
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
# Agent session (background thread → async queue)
# ---------------------------------------------------------------------------

def _run_session(config: dict, prompt: str, max_results: int, gmail_token: dict,
                 q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    def put(item):
        loop.call_soon_threadsafe(q.put_nowait, item)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    session = None
    try:
        session = client.beta.sessions.create(
            agent={"type": "agent", "id": config["agent_id"], "version": config["agent_version"]},
            environment_id=config["environment_id"],
        )

        with client.beta.sessions.events.stream(session_id=session.id) as stream:
            client.beta.sessions.events.send(
                session_id=session.id,
                events=[{"type": "user.message", "content": [{"type": "text", "text": prompt}]}],
            )

            for event in stream:
                if event.type == "agent.message":
                    for block in event.content:
                        if block.type == "text":
                            put({"type": "text", "content": block.text})

                elif event.type == "agent.custom_tool_use" and event.tool_name == "get_recent_emails":
                    tool_input = event.input or {}
                    n = int(tool_input.get("max_results", max_results))
                    fq = tool_input.get("query", "")
                    put({"type": "status", "content": f"Fetching {n} emails" + (f' matching "{fq}"' if fq else "") + "…"})

                    try:
                        emails = fetch_emails(gmail_token, max_results=n, query=fq)
                        result_text = json.dumps(emails, indent=2, ensure_ascii=False)
                    except Exception as exc:
                        result_text = f"Error fetching emails: {exc}"

                    client.beta.sessions.events.send(
                        session_id=session.id,
                        events=[{
                            "type": "user.custom_tool_result",
                            "custom_tool_use_id": event.id,
                            "content": [{"type": "text", "text": result_text}],
                        }],
                    )

                elif event.type == "session.status_idle":
                    if event.stop_reason.type != "requires_action":
                        break

                elif event.type == "session.status_terminated":
                    break

    except Exception as exc:
        put({"type": "error", "content": str(exc)})
    finally:
        if session:
            try:
                client.beta.sessions.delete(session_id=session.id)
            except Exception:
                pass
        put(None)


async def _sse_stream(prompt: str, max_results: int, gmail_token: dict) -> AsyncIterator[str]:
    try:
        config = _load_agent_config()
    except FileNotFoundError as exc:
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    threading.Thread(
        target=_run_session,
        args=(config, prompt, max_results, gmail_token, event_queue, loop),
        daemon=True,
    ).start()

    while True:
        item = await event_queue.get()
        if item is None:
            yield "data: [DONE]\n\n"
            break
        yield f"data: {json.dumps(item)}\n\n"


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/auth/login")
async def auth_login(request: Request):
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    params = urllib.parse.urlencode({
        "client_id": os.environ["GMAIL_CLIENT_ID"],
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    })
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}")


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = None, state: str = None, error: str = None):
    if error:
        return RedirectResponse(f"/?error={error}")
    if not code or state != request.session.get("oauth_state"):
        return HTMLResponse("Invalid OAuth state — please try signing in again.", status_code=400)

    token_resp = http_requests.post(GOOGLE_TOKEN_URL, data={
        "code": code,
        "client_id": os.environ["GMAIL_CLIENT_ID"],
        "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }, timeout=15)
    token_data = token_resp.json()

    if "error" in token_data:
        return HTMLResponse(f"Token error: {token_data.get('error_description', token_data['error'])}", status_code=400)

    user_resp = http_requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
        timeout=10,
    )
    profile = user_resp.json() if user_resp.ok else {}

    request.session["gmail_token"] = {
        "token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "token_uri": GOOGLE_TOKEN_URL,
        "client_id": os.environ["GMAIL_CLIENT_ID"],
        "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
        "scopes": SCOPES.split(),
    }
    request.session["user_name"] = profile.get("name", "")
    request.session["user_email"] = profile.get("email", "")
    request.session["user_picture"] = profile.get("picture", "")
    request.session.pop("oauth_state", None)

    return RedirectResponse("/")


@app.get("/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


# ---------------------------------------------------------------------------
# Main routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, error: str = None):
    if not request.session.get("gmail_token"):
        return _login_page(error)
    return _main_page(
        email=request.session.get("user_email", ""),
        name=request.session.get("user_name", ""),
        picture=request.session.get("user_picture", ""),
    )


@app.get("/summarize")
async def summarize(
    request: Request,
    query: str = Query(default=""),
    max_results: int = Query(default=20, ge=1, le=50),
    prompt: str = Query(default=""),
):
    gmail_token = request.session.get("gmail_token")
    if not gmail_token:
        async def _unauth():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Not signed in. Please refresh and sign in.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_unauth(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    if not prompt:
        prompt = "Please check my email and give me a concise summary of my most recent emails."

    return StreamingResponse(
        _sse_stream(prompt=prompt, max_results=max_results, gmail_token=gmail_token),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------

_SHARED_STYLES = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f7; color: #1d1d1f;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 40px 16px 80px;
  }
  header { text-align: center; margin-bottom: 32px; }
  header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
  header p  { color: #6e6e73; margin-top: 6px; font-size: 0.95rem; }
  .card {
    background: #fff; border-radius: 16px;
    box-shadow: 0 2px 20px rgba(0,0,0,.08);
    padding: 28px; width: 100%; max-width: 720px;
  }
"""


def _login_page(error: str = None) -> str:
    error_html = f'<p class="err">⚠️ {error}</p>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Email Summarizer — Sign In</title>
  <style>
    {_SHARED_STYLES}
    .login-card {{
      background: #fff; border-radius: 20px; padding: 48px 40px;
      box-shadow: 0 2px 20px rgba(0,0,0,.08); text-align: center;
      max-width: 420px; width: 100%;
    }}
    .emoji {{ font-size: 3rem; margin-bottom: 16px; }}
    h2 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }}
    .sub {{ color: #6e6e73; font-size: 0.9rem; margin-bottom: 32px; line-height: 1.5; }}
    .google-btn {{
      display: inline-flex; align-items: center; gap: 12px;
      background: #fff; border: 1.5px solid #d2d2d7; border-radius: 12px;
      padding: 12px 24px; font-size: 0.95rem; font-weight: 600;
      color: #1d1d1f; text-decoration: none;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
      transition: box-shadow .15s, transform .1s;
    }}
    .google-btn:hover {{ box-shadow: 0 2px 10px rgba(0,0,0,.14); transform: translateY(-1px); }}
    .google-btn:active {{ transform: scale(.97); }}
    .err {{ color: #c0392b; font-size: 0.85rem; margin-top: 16px; }}
  </style>
</head>
<body>
<header>
  <h1>📧 Email Summarizer</h1>
  <p>Powered by Claude AI</p>
</header>
<div class="login-card">
  <div class="emoji">📬</div>
  <h2>Sign in to get started</h2>
  <p class="sub">Connect your Gmail account and Claude will summarize<br>your recent emails in seconds.</p>
  <a href="/auth/login" class="google-btn">
    <svg width="20" height="20" viewBox="0 0 48 48">
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
    </svg>
    Sign in with Google
  </a>
  {error_html}
</div>
</body>
</html>"""


def _main_page(email: str, name: str, picture: str) -> str:
    avatar = f'<img src="{picture}" class="avatar" referrerpolicy="no-referrer" />' if picture else ""
    display = name or email
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Email Summarizer</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    {_SHARED_STYLES}
    .topbar {{
      width: 100%; max-width: 720px; display: flex;
      justify-content: flex-end; align-items: center;
      gap: 10px; margin-bottom: 16px;
    }}
    .avatar {{ width: 32px; height: 32px; border-radius: 50%; object-fit: cover; }}
    .user-name {{ font-size: 0.88rem; color: #6e6e73; }}
    .logout-link {{
      font-size: 0.82rem; color: #0071e3; text-decoration: none; font-weight: 500;
    }}
    .logout-link:hover {{ text-decoration: underline; }}
    .controls {{ display: flex; flex-direction: column; gap: 12px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    input[type="text"], input[type="number"] {{
      border: 1.5px solid #d2d2d7; border-radius: 10px;
      padding: 10px 14px; font-size: 0.92rem; outline: none;
      transition: border-color .15s; background: #fafafa;
    }}
    input[type="text"]:focus, input[type="number"]:focus {{ border-color: #0071e3; background: #fff; }}
    #filter {{ flex: 1; min-width: 200px; }}
    #count  {{ width: 90px; }}
    button {{
      border: none; border-radius: 10px; padding: 10px 22px;
      font-size: 0.92rem; font-weight: 600; cursor: pointer;
      transition: opacity .15s, transform .1s;
    }}
    button:active {{ transform: scale(.97); }}
    #btn-check {{ background: #0071e3; color: #fff; flex-shrink: 0; }}
    #btn-check:disabled {{ opacity: .5; cursor: not-allowed; }}
    #btn-copy {{
      background: #f5f5f7; color: #1d1d1f;
      border: 1.5px solid #d2d2d7; font-size: 0.82rem; padding: 6px 14px;
    }}
    .status-bar {{
      display: flex; align-items: center; gap: 10px;
      font-size: 0.85rem; color: #6e6e73; min-height: 24px; margin-top: 4px;
    }}
    .spinner {{
      width: 14px; height: 14px; border: 2px solid #d2d2d7;
      border-top-color: #0071e3; border-radius: 50%;
      animation: spin .7s linear infinite; display: none;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .spinner.active {{ display: block; }}
    hr {{ border: none; border-top: 1.5px solid #f0f0f2; margin: 20px 0; }}
    #output {{ font-size: 0.93rem; line-height: 1.7; color: #1d1d1f; min-height: 40px; }}
    #output.empty {{ color: #aeaeb2; font-style: italic; }}
    #output h1, #output h2, #output h3 {{ font-weight: 700; margin: 16px 0 6px; letter-spacing: -0.3px; }}
    #output h1 {{ font-size: 1.2rem; }} #output h2 {{ font-size: 1.05rem; }} #output h3 {{ font-size: .97rem; }}
    #output p {{ margin: 6px 0; }} #output ul, #output ol {{ padding-left: 20px; margin: 6px 0; }}
    #output li {{ margin: 3px 0; }} #output strong {{ font-weight: 600; }}
    #output code {{ background: #f5f5f7; border-radius: 4px; padding: 1px 5px; font-size: .88em; font-family: "SF Mono", Menlo, monospace; }}
    #output blockquote {{ border-left: 3px solid #d2d2d7; padding-left: 12px; color: #6e6e73; margin: 8px 0; }}
    .error-msg {{ background: #fff2f2; border: 1.5px solid #ffb3b3; border-radius: 10px; padding: 12px 16px; color: #c0392b; font-size: .88rem; }}
    .output-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .output-header span {{ font-size: .8rem; color: #aeaeb2; }}
  </style>
</head>
<body>

<div class="topbar">
  {avatar}
  <span class="user-name">{display}</span>
  <a href="/auth/logout" class="logout-link">Sign out</a>
</div>

<header>
  <h1>📧 Email Summarizer</h1>
  <p>Powered by Claude — your emails, privately summarized</p>
</header>

<div class="card">
  <div class="controls">
    <div class="row">
      <input id="filter" type="text" placeholder='Gmail filter  (e.g. is:unread  or  from:boss@work.com)' />
    </div>
    <div class="row">
      <label style="font-size:.88rem;color:#6e6e73;white-space:nowrap">Emails to fetch:</label>
      <input id="count" type="number" value="20" min="1" max="50" />
      <button id="btn-check" onclick="checkEmail()">Check My Email</button>
    </div>
    <div class="status-bar">
      <div class="spinner" id="spinner"></div>
      <span id="status-text"></span>
    </div>
  </div>

  <hr />

  <div class="output-header" id="output-header" style="display:none">
    <span id="ts"></span>
    <button id="btn-copy" onclick="copyOutput()">Copy</button>
  </div>
  <div id="output" class="empty">Your email summary will appear here.</div>
</div>

<script>
  let rawText = "";

  function checkEmail() {{
    const filter     = document.getElementById("filter").value.trim();
    const maxResults = parseInt(document.getElementById("count").value) || 20;
    const btn        = document.getElementById("btn-check");
    const spinner    = document.getElementById("spinner");
    const statusEl   = document.getElementById("status-text");
    const outputEl   = document.getElementById("output");
    const header     = document.getElementById("output-header");

    btn.disabled = true;
    spinner.classList.add("active");
    statusEl.textContent = "Starting session…";
    outputEl.innerHTML = "";
    outputEl.classList.remove("empty", "error-msg");
    header.style.display = "none";
    rawText = "";

    const params = new URLSearchParams({{ max_results: maxResults }});
    if (filter) params.set("query", filter);

    const src = new EventSource(`/summarize?${{params}}`);

    src.onmessage = (e) => {{
      if (e.data === "[DONE]") {{
        src.close(); btn.disabled = false;
        spinner.classList.remove("active");
        statusEl.textContent = "Done";
        header.style.display = "flex";
        document.getElementById("ts").textContent =
          new Date().toLocaleTimeString([], {{ hour: "2-digit", minute: "2-digit" }});
        return;
      }}
      let msg; try {{ msg = JSON.parse(e.data); }} catch {{ return; }}
      if (msg.type === "text") {{
        rawText += msg.content;
        outputEl.innerHTML = marked.parse(rawText);
      }} else if (msg.type === "status") {{
        statusEl.textContent = msg.content;
      }} else if (msg.type === "error") {{
        src.close(); btn.disabled = false;
        spinner.classList.remove("active"); statusEl.textContent = "";
        outputEl.classList.add("error-msg");
        outputEl.textContent = "⚠️ " + msg.content;
      }}
    }};

    src.onerror = () => {{
      src.close(); btn.disabled = false;
      spinner.classList.remove("active"); statusEl.textContent = "";
      outputEl.classList.add("error-msg");
      outputEl.textContent = "⚠️ Connection error. Is the server running?";
    }};
  }}

  function copyOutput() {{
    navigator.clipboard.writeText(rawText).then(() => {{
      const btn = document.getElementById("btn-copy");
      btn.textContent = "Copied!";
      setTimeout(() => {{ btn.textContent = "Copy"; }}, 1500);
    }});
  }}
</script>
</body>
</html>"""
