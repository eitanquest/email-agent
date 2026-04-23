"""
ONE-TIME SETUP — run once, then save agent_config.json forever.
Run: python setup.py
"""

import anthropic
import json
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

print("Creating environment...")
environment = client.beta.environments.create(
    name="email-agent-env",
    config={
        "type": "cloud",
        "networking": {"type": "unrestricted"},
    },
)

print("Creating agent...")
agent = client.beta.agents.create(
    name="Email Summarizer",
    model="claude-opus-4-7",
    system="""You are a helpful email assistant. When asked to summarize emails:
1. Use the get_recent_emails tool to fetch emails.
2. Organize the summary by priority: urgent items first, then action items, then FYI.
3. Group related threads together.
4. Keep sender names and key dates visible.
5. Be concise — bullet points over paragraphs.""",
    tools=[
        {
            "type": "custom",
            "name": "get_recent_emails",
            "description": (
                "Fetch recent emails from the user's Gmail inbox. "
                "Returns subject, sender, date, and body preview for each email."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "How many emails to fetch (default 20, max 50).",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional Gmail search query, e.g. 'is:unread', "
                            "'after:2024/01/01', 'from:boss@company.com'."
                        ),
                    },
                },
                "required": [],
            },
        }
    ],
)

config = {
    "agent_id": agent.id,
    "agent_version": agent.version,
    "environment_id": environment.id,
}

config_path = os.path.join(os.path.dirname(__file__), "agent_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Agent:       {agent.id}")
print(f"✅ Environment: {environment.id}")
print(f"✅ Config saved to agent_config.json")
print("\nNext: run `python email_agent.py` to check your email.")
