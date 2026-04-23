"""
Email Summarizer — FastAPI web server.

Local dev:
    uvicorn app:app --reload --port 8000

Railway:
    Set env vars (see deploy_prep.py), Railway injects $PORT automatically.
    Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import anthropic
import asyncio
import base64
import json
import os
import threading
from typing import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

app = FastAPI()
DIR = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


# ---------------------------------------------------------------------------
# Config helpers — env vars take priority (Railway), files are local fallback
# ---------------------------------------------------------------------------

def _load_agent_config() -> dict:
    agent_id = os.environ.get("AGENT_ID")
    agent_version = os.environ.get("AGENT_VERSION")
    environment_id = os.environ.get("ENVIRONMENT_ID")

    if agent_id and agent_version and environment_id:
        return {"agent_id": agent_id, "agent_version": agent_version, "environment_id": environment_id}

    config_path = os.path.join(DIR, "agent_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "agent_config.json not found and AGENT_ID/AGENT_VERSION/ENVIRONMENT_ID env vars not set. "
            "Run setup.py first, then run deploy_prep.py to get the Railway env vars."
        )
    with open(config_path) as f:
        return json.load(f)


def _gmail_service():
    """Return authenticated Gmail service. Uses env vars on Railway, token.json locally."""
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")

    if refresh_token and client_id and client_secret:
        # Production path — reconstruct credentials from env vars
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return build("gmail", "v1", credentials=creds)

    # Local dev path — use token.json / credentials.json
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
                    "credentials.json not found. Download it from Google Cloud Console "
                    "or set GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET / GMAIL_REFRESH_TOKEN env vars."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------

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


def fetch_emails(max_results: int = 20, query: str = "") -> list[dict]:
    service = _gmail_service()
    params: dict = {"userId": "me", "maxResults": min(max_results, 50)}
    if query:
        params["q"] = query

    result = service.users().messages().list(**params).execute()
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
# Agent session (runs in a background thread, bridges to async via Queue)
# ---------------------------------------------------------------------------

def _run_session(config: dict, prompt: str, max_results: int, q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    def put(item):
        loop.call_soon_threadsafe(q.put_nowait, item)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    session = None
    try:
        session = client.beta.sessions.create(
            agent={"type": "agent", "id": config["agent_id"], "version": config["agent_version"]},
            environment_id=config["environment_id"],
        )

        with client.beta.sessions.stream(session_id=session.id) as stream:
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
                    label = f"Fetching {n} emails" + (f' matching "{fq}"' if fq else "") + "…"
                    put({"type": "status", "content": label})

                    try:
                        emails = fetch_emails(max_results=n, query=fq)
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


async def _sse_stream(prompt: str, max_results: int) -> AsyncIterator[str]:
    try:
        config = _load_agent_config()
    except FileNotFoundError as exc:
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    thread = threading.Thread(
        target=_run_session,
        args=(config, prompt, max_results, event_queue, loop),
        daemon=True,
    )
    thread.start()

    while True:
        item = await event_queue.get()
        if item is None:
            yield "data: [DONE]\n\n"
            break
        yield f"data: {json.dumps(item)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.get("/summarize")
async def summarize(
    query: str = Query(default=""),
    max_results: int = Query(default=20, ge=1, le=50),
    prompt: str = Query(default=""),
):
    if not prompt:
        prompt = "Please check my email and give me a concise summary of my most recent emails."
    return StreamingResponse(
        _sse_stream(prompt=prompt, max_results=max_results),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Email Summarizer</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f7;
      color: #1d1d1f;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 16px 80px;
    }

    header { text-align: center; margin-bottom: 32px; }
    header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
    header p  { color: #6e6e73; margin-top: 6px; font-size: 0.95rem; }

    .card {
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 2px 20px rgba(0,0,0,.08);
      padding: 28px;
      width: 100%;
      max-width: 720px;
    }

    .controls { display: flex; flex-direction: column; gap: 12px; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

    input[type="text"], input[type="number"] {
      border: 1.5px solid #d2d2d7;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.92rem;
      outline: none;
      transition: border-color .15s;
      background: #fafafa;
    }
    input[type="text"]:focus, input[type="number"]:focus { border-color: #0071e3; background: #fff; }
    #filter { flex: 1; min-width: 200px; }
    #count  { width: 90px; }

    button {
      border: none;
      border-radius: 10px;
      padding: 10px 22px;
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity .15s, transform .1s;
    }
    button:active { transform: scale(.97); }

    #btn-check { background: #0071e3; color: #fff; flex-shrink: 0; }
    #btn-check:disabled { opacity: .5; cursor: not-allowed; }

    #btn-copy {
      background: #f5f5f7;
      color: #1d1d1f;
      border: 1.5px solid #d2d2d7;
      font-size: 0.82rem;
      padding: 6px 14px;
    }

    .status-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.85rem;
      color: #6e6e73;
      min-height: 24px;
      margin-top: 4px;
    }
    .spinner {
      width: 14px; height: 14px;
      border: 2px solid #d2d2d7;
      border-top-color: #0071e3;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner.active { display: block; }

    hr { border: none; border-top: 1.5px solid #f0f0f2; margin: 20px 0; }

    #output { font-size: 0.93rem; line-height: 1.7; color: #1d1d1f; min-height: 40px; }
    #output.empty { color: #aeaeb2; font-style: italic; }

    #output h1, #output h2, #output h3 { font-weight: 700; margin: 16px 0 6px; letter-spacing: -0.3px; }
    #output h1 { font-size: 1.2rem; }
    #output h2 { font-size: 1.05rem; }
    #output h3 { font-size: 0.97rem; }
    #output p  { margin: 6px 0; }
    #output ul, #output ol { padding-left: 20px; margin: 6px 0; }
    #output li { margin: 3px 0; }
    #output strong { font-weight: 600; }
    #output code {
      background: #f5f5f7; border-radius: 4px;
      padding: 1px 5px; font-size: 0.88em;
      font-family: "SF Mono", Menlo, monospace;
    }
    #output blockquote { border-left: 3px solid #d2d2d7; padding-left: 12px; color: #6e6e73; margin: 8px 0; }

    .error-msg {
      background: #fff2f2; border: 1.5px solid #ffb3b3;
      border-radius: 10px; padding: 12px 16px;
      color: #c0392b; font-size: 0.88rem;
    }

    .output-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .output-header span { font-size: 0.8rem; color: #aeaeb2; }
  </style>
</head>
<body>

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

  function checkEmail() {
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

    const params = new URLSearchParams({ max_results: maxResults });
    if (filter) params.set("query", filter);

    const src = new EventSource(`/summarize?${params}`);

    src.onmessage = (e) => {
      if (e.data === "[DONE]") {
        src.close();
        btn.disabled = false;
        spinner.classList.remove("active");
        statusEl.textContent = "Done";
        header.style.display = "flex";
        document.getElementById("ts").textContent =
          new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return;
      }
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }

      if (msg.type === "text") {
        rawText += msg.content;
        outputEl.innerHTML = marked.parse(rawText);
      } else if (msg.type === "status") {
        statusEl.textContent = msg.content;
      } else if (msg.type === "error") {
        src.close();
        btn.disabled = false;
        spinner.classList.remove("active");
        statusEl.textContent = "";
        outputEl.classList.add("error-msg");
        outputEl.textContent = "⚠️ " + msg.content;
      }
    };

    src.onerror = () => {
      src.close();
      btn.disabled = false;
      spinner.classList.remove("active");
      statusEl.textContent = "";
      outputEl.classList.add("error-msg");
      outputEl.textContent = "⚠️ Connection error. Is the server running?";
    };
  }

  function copyOutput() {
    navigator.clipboard.writeText(rawText).then(() => {
      const btn = document.getElementById("btn-copy");
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
  }
</script>
</body>
</html>
"""
