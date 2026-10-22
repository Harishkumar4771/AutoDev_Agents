# 🤖 AutoDev Agents

A **multi-agent coding pipeline** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Streamlit](https://streamlit.io/).  
It takes a coding requirement, writes code and tests, runs them in an isolated sandbox, and loops until tests pass — all orchestrated as a LangGraph `StateGraph`.

## 👥 The 6 Agents

| # | Agent | Role | Implementation |
|---|-------|------|----------------|
| **0** | **Code Generation Agent** 🆕 | Synthesises code, tests, and task breakdown from natural language requirements | Ollama LLM (`codellama`) + Smart Keyword Fallback |
| **1** | **Planner** | Outlines tasks, function signature, and edge cases | LangGraph Node |
| **2** | **Developer** | Writes initial code draft and applies patches based on Critic diagnosis | Canned preset / LLM patch on retry |
| **3** | **Test Agent** | Generates a 5-test pytest suite (strictly blind to the code) | Independent test generator |
| **4** | **Executor** | Writes code + tests to a temporary directory and executes `pytest` | Isolated subprocess runner with timeout |
| **5** | **Critic** | Runs `ast` + `flake8` static checks and inspects pytest output to decide **pass**, **retry**, or **escalate** | Static Analysis + Test Judge hard gate |

### Pipeline Flow

```
                    START
                      │
                      ▼
          [0] Code Generation Agent
             (LLM / Fallback Synth)
                      │
                      ▼
              [1] Planner Agent
             (Tasks & Edge Cases)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    [2] Developer            [3] Test Agent
     (Code Draft)          (Blind Pytest Suite)
          │                       │
          └───────────┬───────────┘
                      ▼
              [4] Executor Agent
             (Sandbox Pytest Run)
                      │
                      ▼
               [5] Critic Agent
           (AST + Flake8 + Pytest)
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
         retry       pass     escalate
           │          │          │
     [2] Developer   END        END
      (Patch code)
```

## 🧠 Hybrid Engine: LLM + Deterministic Presets

1. **Preset Requirements (Deterministic Benchmark):**
   - **Second-largest unique number** — v1 fails on duplicates; v2 patches with deduplication.
   - **Validate Indian mobile number** — v1 fails on `+91`/`91` prefix; v2 regex fix.
   - **Palindrome check (ignore case & punctuation)** — v1 fails on punctuation; v2 cleans symbols.

2. **Custom Requirement Mode (Real Code Generation):**
   - Enter any custom Python coding task in natural language.
   - **Code Generation Agent** connects to local **Ollama** (`codellama` / `llama3`) or uses smart keyword synthesis when offline.
   - The generated code is put through the same Planner → Test Agent → Executor → Critic test-fail-fix loop!

## ⚡ Agent Health Check Dashboard

The dashboard provides a real-time status matrix showing every agent's:
- **Operational status** (OK, Retry, Escalate, Skipped, LLM)
- **Execution latency** (in milliseconds)
- **Detailed diagnostic message**

## 🚀 Run Locally

```bash
# 1. Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Start Ollama for real local LLM generation
ollama run codellama

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select a preset or **Custom Requirement** and press **🚀 Run Agents**.

## 🛠️ Tech Stack

- **Streamlit** — Interactive dark-themed UI with live terminal logs
- **LangGraph** — `StateGraph` with conditional edges for the retry loop
- **pytest** — Isolated sandbox test execution
- **flake8 & ast** — Static syntax and linter checks
- **Ollama** — Optional local LLM endpoint for autonomous code generation
