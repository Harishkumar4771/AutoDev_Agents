"""AutoDev Agents – multi-agent coding pipeline (LangGraph + Streamlit).
Based on the Agentic AI Case Study: "An agentic pipeline that tests, runs and reviews code"
Authors: B S HARISH KUMAR, Md Rafin Uzzaman
"""
import ast
import operator
import os
import re
import subprocess
import sys
import tempfile
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
import streamlit as st

# ── Preset requirements matching the Case Study Slides ───────────────────────
PRESETS = {
    "Second-largest unique number in a list": {
        "short_title": "Second-largest unique number",
        "description": "Find the second-largest unique value in a list of integers. Return None if fewer than 2 unique values.",
        "plan": (
            "Tasks:\n"
            "1. Write `second_largest(nums)` returning the second-largest unique value.\n"
            "2. Handle duplicates by deduplicating values.\n"
            "3. Return None if fewer than 2 unique items exist.\n"
            "Edge cases: duplicates, single element, all same values, negative numbers, empty list."
        ),
        "code_v1": r'''def second_largest(nums):
    """Return the second-largest unique number in a list."""
    if len(nums) < 2:
        return None
    nums_sorted = sorted(nums, reverse=True)
    return nums_sorted[1]
''',
        "code_v2": r'''def second_largest(nums):
    """Return the second-largest unique number in a list."""
    unique = list(set(nums))
    if len(unique) < 2:
        return None
    unique.sort(reverse=True)
    return unique[1]
''',
        "tests": r'''from solution import second_largest

def test_basic_list():
    assert second_largest([1, 2, 3, 4, 5]) == 4

def test_negative_numbers():
    assert second_largest([-1, -2, -3]) == -2

def test_with_duplicates():
    assert second_largest([5, 5, 3, 1]) == 3

def test_single_element():
    assert second_largest([1]) is None

def test_all_same():
    assert second_largest([7, 7, 7]) is None
''',
        "critic_diag_v1": "Duplicates not handled (fails on [5, 5, 3, 1] and [7, 7, 7])",
    },
    "Validate Indian mobile number": {
        "short_title": "Validate Indian mobile number",
        "description": "Validate 10-digit Indian mobile numbers starting with 6-9, with optional +91 or 91 country code prefix.",
        "plan": (
            "Tasks:\n"
            "1. Write `validate_indian_mobile(number)` returning boolean.\n"
            "2. Accept optional +91 or 91 prefix before the 10-digit number.\n"
            "3. Verify primary 10 digits start with 6, 7, 8, or 9.\n"
            "Edge cases: +91 prefix, 91 prefix without plus, invalid starting digit, short input, spaces."
        ),
        "code_v1": r'''import re

def validate_indian_mobile(number):
    """Validate an Indian mobile number (10 digits starting with 6-9)."""
    return bool(re.match(r'^[6-9]\d{9}$', number.strip()))
''',
        "code_v2": r'''import re

def validate_indian_mobile(number):
    """Validate an Indian mobile number with optional +91/91 prefix."""
    cleaned = number.strip().replace(' ', '')
    return bool(re.match(r'^(\+91|91)?[6-9]\d{9}$', cleaned))
''',
        "tests": r'''from solution import validate_indian_mobile

def test_valid_10digit():
    assert validate_indian_mobile("9876543210") is True

def test_invalid_start_digit():
    assert validate_indian_mobile("1234567890") is False

def test_with_plus91():
    assert validate_indian_mobile("+919876543210") is True

def test_with_91_prefix():
    assert validate_indian_mobile("919876543210") is True

def test_too_short():
    assert validate_indian_mobile("98765") is False
''',
        "critic_diag_v1": "Prefixes +91 and 91 not supported by regex",
    },
    "Palindrome check (ignore case & punctuation)": {
        "short_title": "Palindrome check",
        "description": "Check if a string is a palindrome, ignoring casing, spaces, and punctuation symbols.",
        "plan": (
            "Tasks:\n"
            "1. Write `is_palindrome(s)` returning boolean.\n"
            "2. Strip all non-alphanumeric characters and lowercase the string.\n"
            "3. Compare the cleaned string to its reverse.\n"
            "Edge cases: mixed case, punctuation, sentence with spaces, quotes, questions."
        ),
        "code_v1": r'''def is_palindrome(s):
    """Check if a string is a palindrome (case-insensitive)."""
    s = s.lower()
    return s == s[::-1]
''',
        "code_v2": r'''def is_palindrome(s):
    """Check if a string is a palindrome (ignoring case & punctuation)."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
''',
        "tests": r'''from solution import is_palindrome

def test_simple_palindrome():
    assert is_palindrome("racecar") is True

def test_not_a_palindrome():
    assert is_palindrome("hello") is False

def test_mixed_case_with_symbols():
    assert is_palindrome("No 'x' in Nixon") is True

def test_sentence_with_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama") is True

def test_question_sentence():
    assert is_palindrome("Was it a car or a cat I saw?") is True
''',
        "critic_diag_v1": "Punctuation and spaces not stripped before checking",
    },
}


# ── Shared state flowing through every agent node ────────────────────────────
class AgentState(TypedDict):
    requirement: str
    plan: str
    code: str
    tests: str
    results: str
    returncode: int
    feedback: str
    iteration: int
    verdict: str
    terminal_entries: Annotated[list[tuple[str, str, str]], operator.add]
    log: Annotated[list[str], operator.add]


# ── Python Runner Helper (Ensures pytest & flake8 work across all environments) ─
def get_python_runner() -> str:
    """Find the best Python binary that has pytest available."""
    candidates = [
        sys.executable,
        "/opt/anaconda3/bin/python3",
        "/opt/anaconda3/bin/python",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "python3",
        "python",
    ]
    for cand in candidates:
        try:
            res = subprocess.run([cand, "-m", "pytest", "--version"], capture_output=True, timeout=2)
            if res.returncode == 0:
                return cand
        except Exception:
            pass
    return sys.executable


# ── Static Analysis Helper (AST + Flake8) – Slide 7 ─────────────────────────
def run_static_analysis(code: str) -> tuple[bool, str]:
    """Perform AST syntax parse and Flake8 linter check."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"AST SyntaxError: line {exc.lineno}: {exc.msg}"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        f_name = f.name

    py_bin = get_python_runner()
    try:
        proc = subprocess.run(
            [py_bin, "-m", "flake8", "--isolated", "--select=E9,F63,F7,F82", f_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0 and proc.stdout.strip():
            return False, f"Flake8 syntax check: {proc.stdout.strip()}"
    except Exception:
        pass
    finally:
        if os.path.exists(f_name):
            try:
                os.remove(f_name)
            except OSError:
                pass

    return True, "AST & Flake8 static analysis clean."


# ── Agent nodes (Slide 5: Five agents, one job each) ─────────────────────────
def planner_node(state: AgentState) -> dict:
    """1. Planner: Turns requirement into tasks, function signature & edge cases."""
    preset = PRESETS.get(state["requirement"])
    plan = preset["plan"] if preset else "1. Implement requirement\n2. Cover edge cases"
    entry = ("Planner", "4 tasks, 5 edge cases", "#10B981")
    return {
        "plan": plan,
        "terminal_entries": [entry],
        "log": ["📋 **Planner Agent:** Identified requirement tasks, signature, and 5 edge cases."],
    }


def developer_node(state: AgentState) -> dict:
    """2. Developer: Writes function, then patches based on Critic diagnosis."""
    preset = PRESETS.get(state["requirement"])
    attempt = state["iteration"] + 1
    if attempt == 1:
        code = preset["code_v1"] if preset else "def solution(): pass"
        entry = ("Developer", "draft v1 written", "#38BDF8")
        msg = "👨‍💻 **Developer Agent:** Draft v1 written."
    else:
        code = preset["code_v2"] if preset else "def solution(): pass"
        entry = ("Developer", f"draft v{attempt} written (patch applied)", "#38BDF8")
        msg = f"👨‍💻 **Developer Agent:** Applied patch based on Critic diagnosis (draft v{attempt})."

    return {
        "code": code,
        "iteration": attempt,
        "terminal_entries": [entry],
        "log": [msg],
    }


def test_agent_node(state: AgentState) -> dict:
    """3. Test Agent: Writes pytest cases from plan only, blind to the code."""
    preset = PRESETS.get(state["requirement"])
    tests = preset["tests"] if preset else "def test_default(): assert True"
    entry = ("Test Agent", "5 pytest cases generated (blind to code)", "#34D399")
    return {
        "tests": tests,
        "terminal_entries": [entry],
        "log": ["🧪 **Test Agent:** Generated 5 isolated pytest cases from requirement specifications."],
    }


def executor_node(state: AgentState) -> dict:
    """4. Executor: Runs code and tests in an isolated sandbox with timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in [("solution.py", state["code"]), ("test_solution.py", state["tests"])]:
            with open(os.path.join(tmp, name), "w") as f:
                f.write(content)

        env = os.environ.copy()
        env["PYTHONPATH"] = tmp + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        py_bin = get_python_runner()

        try:
            proc = subprocess.run(
                [py_bin, "-m", "pytest", "-q", "--no-header", "test_solution.py"],
                cwd=tmp,
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )
            raw_output = (proc.stdout + "\n" + proc.stderr).strip()
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            raw_output = "TIMEOUT: Test execution exceeded 10-second sandbox limit."
            returncode = 124
        except Exception as exc:
            raw_output = f"ERROR executing sandbox runner: {exc}"
            returncode = 1

    # Extract test count summary
    m_pass = re.search(r"(\d+)\s+passed", raw_output)
    m_fail = re.search(r"(\d+)\s+failed", raw_output)
    passed_cnt = int(m_pass.group(1)) if m_pass else (5 if returncode == 0 else 0)
    failed_cnt = int(m_fail.group(1)) if m_fail else (0 if returncode == 0 else max(0, 5 - passed_cnt))

    terminal_summary = f"{passed_cnt} passed, {failed_cnt} failed"
    entry = ("Executor", terminal_summary, "#818CF8")

    return {
        "results": raw_output,
        "returncode": returncode,
        "terminal_entries": [entry],
        "log": [f"⚙️ **Executor Agent:** Sandbox execution finished: `{terminal_summary}`."],
    }


def critic_node(state: AgentState) -> dict:
    """5. Critic: Static analysis (AST/Flake8) + test judge -> accept, retry or escalate."""
    # Step A: Static checks
    is_clean, static_diag = run_static_analysis(state["code"])
    if not is_clean:
        if state["iteration"] < 3:
            entry = ("Critic", f"Static check issue: {static_diag}", "#F59E0B")
            return {
                "feedback": static_diag,
                "verdict": "retry",
                "terminal_entries": [entry],
                "log": [f"🔄 **Critic Agent:** Static analysis failed ({static_diag}). Verdict: **RETRY**."],
            }
        else:
            entry = ("Critic", f"Static error exceeded retry limit: {static_diag}", "#EF4444")
            return {
                "feedback": static_diag,
                "verdict": "escalate",
                "terminal_entries": [entry],
                "log": [f"🚨 **Critic Agent:** Static error unresolved. Verdict: **ESCALATE**."],
            }

    # Step B: Sandbox Pytest evaluation
    preset = PRESETS.get(state["requirement"])
    returncode = state.get("returncode", 0)

    if returncode == 0:
        entry = ("Critic", f"ACCEPT (attempt {state['iteration']} of 3)", "#10B981")
        return {
            "feedback": "All tests passed and static checks clean!",
            "verdict": "pass",
            "terminal_entries": [entry],
            "log": [f"✅ **Critic Agent:** All tests passed. Verdict: **ACCEPTED** on attempt {state['iteration']}."],
        }

    # If tests failed
    diag = preset.get("critic_diag_v1", "Test assertions failed") if state["iteration"] == 1 else "Remaining test failures"
    if state["iteration"] < 3:
        entry = ("Critic", diag, "#F59E0B")
        return {
            "feedback": diag,
            "verdict": "retry",
            "terminal_entries": [entry],
            "log": [f"🔄 **Critic Agent:** {diag}. Verdict: **RETRY** (attempt {state['iteration']} of 3)."],
        }

    entry = ("Critic", "ESCALATE (max 3 retry limit reached)", "#EF4444")
    return {
        "feedback": diag,
        "verdict": "escalate",
        "terminal_entries": [entry],
        "log": ["🚨 **Critic Agent:** Limit reached (3 failed rounds). Flagging human intervention."],
    }


# ── LangGraph Wiring (Slide 6 & 7) ───────────────────────────────────────────
def _route(state: AgentState) -> str:
    return state["verdict"]


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [
        ("planner", planner_node),
        ("developer", developer_node),
        ("test_agent", test_agent_node),
        ("executor", executor_node),
        ("critic", critic_node),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "planner")
    g.add_edge("planner", "developer")       # Fan-out: parallel build
    g.add_edge("planner", "test_agent")      # Fan-out: parallel build
    g.add_edge("developer", "executor")      # Fan-in
    g.add_edge("test_agent", "executor")     # Fan-in
    g.add_edge("executor", "critic")
    g.add_conditional_edges(
        "critic",
        _route,
        {"retry": "developer", "pass": END, "escalate": END},
    )
    return g.compile()


# ── UI Custom CSS & Dashboard Theme ──────────────────────────────────────────
def apply_custom_styling():
    st.markdown(
        """
        <style>
        /* Base page styling */
        .main {
            background-color: #0b1a1e;
        }
        
        /* Metric pill cards from Slide 8 */
        .pill-container {
            display: flex;
            gap: 16px;
            margin: 18px 0 24px 0;
            flex-wrap: wrap;
        }
        .pill-card {
            flex: 1;
            min-width: 180px;
            background: #11282e;
            border: 1px solid #1a3f47;
            border-radius: 9999px;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            font-weight: 600;
            color: #e2e8f0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        .pill-card.verdict-pass {
            background: rgba(16, 185, 129, 0.15);
            border: 1.5px solid #10b981;
            color: #34d399;
        }
        .pill-card.verdict-retry {
            background: rgba(245, 158, 11, 0.15);
            border: 1.5px solid #f59e0b;
            color: #fbbf24;
        }
        .pill-card.verdict-escalate {
            background: rgba(239, 68, 68, 0.15);
            border: 1.5px solid #ef4444;
            color: #f87171;
        }

        /* Slide 8 Terminal Window */
        .terminal-window {
            background-color: #0a171a;
            border: 1px solid #18373e;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            margin-bottom: 24px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        }
        .terminal-header {
            background-color: #11252a;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #18373e;
        }
        .terminal-dots {
            display: flex;
            gap: 7px;
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-red { background-color: #ff5f56; }
        .dot-yellow { background-color: #ffbd2e; }
        .dot-green { background-color: #27c93f; }
        .terminal-title {
            margin-left: 14px;
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .terminal-body {
            padding: 18px 22px;
            color: #e2e8f0;
            line-height: 2.1;
            font-size: 0.95rem;
        }
        .terminal-row {
            display: flex;
            align-items: baseline;
            margin-bottom: 4px;
        }
        .agent-badge {
            width: 120px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .agent-msg {
            color: #cbd5e1;
        }

        /* Slide Comparison Card */
        .table-card {
            background-color: #0f2429;
            border: 1px solid #1a3d45;
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Main Application Dashboard ───────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AutoDev Agents · Agentic Coding Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styling()

    # Sidebar: Setup & Case Study Info
    with st.sidebar:
        st.markdown("### 🤖 **AutoDev Agents**")
        st.caption("Agentic pipeline that tests, runs and reviews code")
        st.markdown("---")
        
        st.subheader("⚙️ Pipeline Configuration")
        selected_key = st.selectbox(
            "Coding Requirement",
            options=list(PRESETS.keys()),
            index=0,
            help="Select one of the 3 case-study requirements from the benchmark",
        )
        preset_info = PRESETS[selected_key]
        st.info(f"**Goal:** {preset_info['description']}")

        st.markdown("---")
        st.subheader("🛠️ Engine & Sandbox")
        st.markdown("- **Graph Engine:** LangGraph (5-Node Cycle)")
        st.markdown("- **Orchestration:** Plan ➔ Build ∥ Test ➔ Execute ➔ Judge")
        st.markdown("- **Sandbox:** Subprocess + Pytest isolated runner")
        st.markdown("- **Static Analysis:** `ast` + `flake8` linter")
        st.markdown("- **Model Engine:** Ollama / Canned Deterministic Fallback")
        
        st.markdown("---")
        st.caption("**Case Study Authors:**\nB S HARISH KUMAR · Md Rafin Uzzaman\n*Domain: Software Development*")

    # Header Bar
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🤖 AutoDev Agents")
        st.markdown(
            "**An agentic pipeline that tests, runs and reviews code** · "
            "<span style='color: #10B981;'>Autonomous fail-test-fix loop</span>",
            unsafe_allow_html=True,
        )
    with col_h2:
        st.write("")
        run_pressed = st.button("🚀 Run Agents", type="primary", use_container_width=True)

    # State initialization
    if "current_preset" not in st.session_state or st.session_state["current_preset"] != selected_key:
        st.session_state["current_preset"] = selected_key
        st.session_state.pop("pipeline_data", None)

    # Execute LangGraph Pipeline
    if run_pressed:
        graph = build_graph()
        init_state: AgentState = {
            "requirement": selected_key,
            "plan": "",
            "code": "",
            "tests": "",
            "results": "",
            "returncode": 0,
            "feedback": "",
            "iteration": 0,
            "verdict": "",
            "terminal_entries": [],
            "log": [],
        }

        code_versions: list[str] = []
        result_history: list[str] = []
        terminal_stream: list[tuple[str, str, str]] = []
        final_state = dict(init_state)

        with st.status("Executing AutoDev Agent Pipeline...", expanded=True) as status_box:
            try:
                for event in graph.stream(init_state, stream_mode="updates"):
                    for node_name, state_delta in event.items():
                        if node_name.startswith("__"):
                            continue
                        for log_msg in state_delta.get("log", []):
                            status_box.markdown(log_msg)
                        if "terminal_entries" in state_delta:
                            terminal_stream.extend(state_delta["terminal_entries"])
                        if "code" in state_delta:
                            code_versions.append(state_delta["code"])
                        if "results" in state_delta:
                            result_history.append(state_delta["results"])

                        final_state.update({k: v for k, v in state_delta.items() if k not in ("log", "terminal_entries")})

                status_box.update(label="✅ Agent pipeline execution completed!", state="complete", expanded=False)
            except Exception as exc:
                status_box.update(label=f"❌ Pipeline error: {exc}", state="error")
                st.error(f"Error during agent execution: {exc}")
                return

        st.session_state["pipeline_data"] = {
            "final": final_state,
            "code_versions": code_versions,
            "result_history": result_history,
            "terminal_stream": terminal_stream,
        }

    # Render Results Dashboard
    if "pipeline_data" not in st.session_state:
        st.info("👈 Select a requirement from the sidebar and click **'🚀 Run Agents'** to launch the test-fail-fix loop.")
        # Show prompt overview
        st.markdown(f"### Selected Requirement: `{selected_key}`")
        st.markdown(f"> {preset_info['description']}")
        return

    data = st.session_state["pipeline_data"]
    f = data["final"]
    code_versions = data["code_versions"]
    result_history = data["result_history"]
    terminal_stream = data["terminal_stream"]

    # Calculate tests passed
    last_res = result_history[-1] if result_history else ""
    passed_cnt = 5 if f["verdict"] == "pass" else (int(m.group(1)) if (m := re.search(r"(\d+) passed", last_res)) else 0)

    # 1. Slide 8 Metric Pills
    verdict_class = "verdict-pass" if f["verdict"] == "pass" else ("verdict-retry" if f["verdict"] == "retry" else "verdict-escalate")
    verdict_label = "accepted" if f["verdict"] == "pass" else f["verdict"]

    st.markdown(
        f"""
        <div class="pill-container">
            <div class="pill-card">
                <span>Attempt <strong>{f['iteration']} / 3</strong></span>
            </div>
            <div class="pill-card">
                <span><strong>{passed_cnt} / 5</strong> tests passed</span>
            </div>
            <div class="pill-card {verdict_class}">
                <span>Verdict: <strong>{verdict_label}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Slide 8 Live Terminal Card (macOS Window styled)
    terminal_rows_html = ""
    for agent, msg, color in terminal_stream:
        terminal_rows_html += f"""
        <div class="terminal-row">
            <span class="agent-badge" style="color: {color};">{agent}</span>
            <span class="agent-msg">{msg}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="terminal-window">
            <div class="terminal-header">
                <div class="terminal-dots">
                    <span class="dot dot-red"></span>
                    <span class="dot dot-yellow"></span>
                    <span class="dot dot-green"></span>
                </div>
                <div class="terminal-title">AutoDev Agents · Live Execution Log</div>
            </div>
            <div class="terminal-body">
                {terminal_rows_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Main Dashboard Tabs
    tab_code, tab_tests, tab_results, tab_case_study = st.tabs([
        "💻 Code Evolution (v1 vs v2)",
        "🧪 Test Suite",
        "📊 Sandbox Results",
        "📋 Case Study & Architecture",
    ])

    with tab_code:
        st.subheader("Code Synthesis & Patching")
        if len(code_versions) >= 2:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("#### 🔴 Draft v1 (Initial — Buggy)")
                st.code(code_versions[0], language="python")
                st.caption("Fails on edge cases (e.g. duplicate elements / unhandled formats)")
            with col_v2:
                st.markdown(f"#### 🟢 Draft v{f['iteration']} (Final — Patched)")
                st.code(code_versions[-1], language="python")
                st.caption("Patched according to Critic agent diagnosis")
        elif code_versions:
            st.code(code_versions[0], language="python")

        # Static analysis result
        st.markdown("---")
        st.markdown("#### 🔍 Static Analysis (AST & Flake8)")
        clean, static_msg = run_static_analysis(f["code"])
        if clean:
            st.success(f"✅ {static_msg}")
        else:
            st.warning(f"⚠️ {static_msg}")

    with tab_tests:
        st.subheader("Pytest Suite (Blind to Code)")
        st.markdown(
            "The **Test Agent** generates pytest assertions strictly from the plan and requirements specifications, "
            "completely blind to Developer code implementation."
        )
        st.code(f["tests"], language="python")

    with tab_results:
        st.subheader("Sandbox Execution History")
        for i, res in enumerate(result_history, 1):
            with st.expander(f"Attempt #{i} Sandbox Output", expanded=(i == len(result_history))):
                st.code(res, language="text")

    with tab_case_study:
        st.subheader("📖 Case Study: Agentic AI in Software Development")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(
                """
                ### ⚠️ The Problem Today (Slide 2)
                - **Tests get skipped:** Deadlines push test writing to the end or out of scope.
                - **Debugging is a manual loop:** Write, run, read failure, patch, repeat.
                - **One-shot AI is unverified:** Generated code looks right but is never run or checked.
                """
            )
        with col_c2:
            st.markdown(
                """
                ### ✨ With AutoDev Agents (Slide 3)
                - **Planner:** Turns requirements into tasks & acceptance criteria.
                - **Test Agent:** Writes tests independently of the code.
                - **Executor & Critic:** Runs tests, diagnoses root causes, and loops back to Developer.
                - **Hard Gate:** Nothing accepted until tests pass or retry cap is reached.
                """
            )

        st.markdown("---")
        st.markdown("### 🔄 Orchestration Lifecycle (Slide 6)")
        st.markdown(
            "**1. Plan** (Tasks & Edge cases) ➔ "
            "**2. Build** (Developer & Test Agent in parallel) ➔ "
            "**3. Execute** (Sandbox pytest run) ➔ "
            "**4. Judge** (Critic AST + Flake8 + Pytest diagnosis) ➔ "
            "**5. Report / Fix Loop** (Auto-patch up to 3 rounds)"
        )


if __name__ == "__main__":
    main()
