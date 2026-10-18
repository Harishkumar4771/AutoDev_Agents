# 🤖 AutoDev Agents

A **multi-agent coding pipeline** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Streamlit](https://streamlit.io/).  
It takes a coding requirement, writes code and tests, runs them, and loops until the tests pass — all orchestrated as a LangGraph `StateGraph`.

## How It Works

| Agent | Role |
|-------|------|
| **Planner** | Outlines tasks and edge cases for the selected requirement |
| **Developer** | Writes code (v1 on first attempt; applies a fix on retry) |
| **Test Agent** | Generates a 5-test pytest suite (independent of the code) |
| **Executor** | Writes code + tests to a temp dir and runs `pytest -q` for real |
| **Critic** | Runs `ast.parse` as a static check, then inspects pytest results to decide: **pass**, **retry**, or **escalate** |

### Pipeline Flow

```
Planner ──┬──→ Developer  ──┐
          └──→ Test Agent ──┘──→ Executor ──→ Critic
                                                │
                              ┌─────────────────┤
                              ▼                  ▼
                          Developer           END
                         (retry loop)     (pass / escalate)
```

## ⚠️ Mocked LLM

There are **no API keys, no Ollama, no network calls**.  
The "LLM" is a Python dictionary of canned outputs, so the fail → fix → pass loop is **fully deterministic**:

1. **Attempt 1** — Developer returns *buggy* code (v1), some tests fail, Critic explains why and retries.
2. **Attempt 2** — Developer returns the *fixed* code (v2), all tests pass, Critic verdict: **PASS**.

## Preset Requirements

1. **Second-largest unique number** — v1 fails on duplicates
2. **Validate Indian mobile number** — v1 fails on `+91` prefix
3. **Palindrome check (ignore case & punctuation)** — v1 fails on punctuation

## Run Locally

```bash
# 1. Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select a preset and press **Run Agents**.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and deploy the repo.
3. Set the main file to `app.py` — no secrets or environment variables needed.

## Tech Stack

- **Streamlit** — interactive UI with live streaming
- **LangGraph** — `StateGraph` with conditional edges for the retry loop
- **pytest** — real test execution in a sandboxed temp directory
- **Python 3.11**
