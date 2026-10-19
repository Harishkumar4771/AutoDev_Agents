"""AutoDev Agents – multi-agent coding pipeline (LangGraph + Streamlit, mocked LLM)."""
import ast, operator, os, re, subprocess, sys, tempfile
from typing import Annotated, TypedDict
import streamlit as st
from langgraph.graph import END, START, StateGraph

# ── Mock-LLM bank: canned plan / code / tests for 3 presets ─────────────────
PRESETS = {
    "Second-largest unique number": {
        "plan": ("Tasks:\n1. Write `second_largest(nums)` → second-largest *unique* value.\n"
                 "2. Return None if fewer than 2 unique values.\n"
                 "Edge cases: duplicates, single element, all-same, negatives."),
        "code_v1": """\
def second_largest(nums):
    '''Return the second-largest unique number in a list.'''
    if len(nums) < 2:
        return None
    nums_sorted = sorted(nums, reverse=True)
    return nums_sorted[1]
""",
        "code_v2": """\
def second_largest(nums):
    '''Return the second-largest unique number in a list.'''
    unique = list(set(nums))
    if len(unique) < 2:
        return None
    unique.sort(reverse=True)
    return unique[1]
""",
        "tests": """\
from solution import second_largest

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
""",
    },
    "Validate Indian mobile number": {
        "plan": ("Tasks:\n1. Write `validate_indian_mobile(number)` → True / False.\n"
                 "2. Accept optional +91 or 91 prefix before the 10-digit number.\n"
                 "Edge cases: +91 prefix, 91 prefix, wrong start digit, too-short input."),
        "code_v1": """\
import re

def validate_indian_mobile(number):
    '''Validate an Indian mobile number (10 digits starting with 6-9).'''
    return bool(re.match(r'^[6-9]\d{9}$', number.strip()))
""",
        "code_v2": """\
import re

def validate_indian_mobile(number):
    '''Validate an Indian mobile number with optional +91/91 prefix.'''
    cleaned = number.strip().replace(' ', '')
    return bool(re.match(r'^(\+91|91)?[6-9]\d{9}$', cleaned))
""",
        "tests": """\
from solution import validate_indian_mobile

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
""",
    },
    "Palindrome check (ignore case & punctuation)": {
        "plan": ("Tasks:\n1. Write `is_palindrome(s)` → True / False.\n"
                 "2. Strip non-alphanumeric characters and ignore case.\n"
                 "Edge cases: mixed case, spaces, apostrophes, colons, question marks."),
        "code_v1": """\
def is_palindrome(s):
    '''Check if a string is a palindrome (case-insensitive).'''
    s = s.lower()
    return s == s[::-1]
""",
        "code_v2": """\
def is_palindrome(s):
    '''Check if a string is a palindrome (ignoring case & punctuation).'''
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
""",
        "tests": """\
from solution import is_palindrome

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
""",
    },
}

# ── Shared state flowing through every agent node ────────────────────────────
class AgentState(TypedDict):
    requirement: str; plan: str; code: str; tests: str  # noqa: E702
    results: str; feedback: str; iteration: int; verdict: str  # noqa: E702
    log: Annotated[list[str], operator.add]

# ── Agent nodes ──────────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    plan = PRESETS[state["requirement"]]["plan"]
    return {"plan": plan, "log": ["📋 **Planner:** identified tasks and edge cases."]}

def developer_node(state: AgentState) -> dict:
    preset, attempt = PRESETS[state["requirement"]], state["iteration"] + 1
    if attempt == 1:
        code, msg = preset["code_v1"], "👨‍💻 **Developer:** wrote initial code (v1)."
    else:
        code, msg = preset["code_v2"], f"👨‍💻 **Developer:** applied fix (v{attempt})."
    return {"code": code, "iteration": attempt, "log": [msg]}

def test_agent_node(state: AgentState) -> dict:
    """Return the test suite – does NOT read code from state."""
    return {"tests": PRESETS[state["requirement"]]["tests"],
            "log": ["🧪 **Test Agent:** generated 5 pytest tests."]}

def executor_node(state: AgentState) -> dict:
    """Write code + tests to a temp dir, run pytest -q, capture output."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in [("solution.py", state["code"]), ("test_solution.py", state["tests"])]:
            with open(os.path.join(tmp, name), "w") as f:
                f.write(content)
        try:
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                                 cwd=tmp, capture_output=True, text=True, timeout=10)
            output = (proc.stdout + proc.stderr).strip()
        except subprocess.TimeoutExpired:
            output = "TIMEOUT: tests exceeded 10-second limit."
        except Exception as exc:
            output = f"ERROR running pytest: {exc}"
    return {"results": output, "log": [f"⚙️ **Executor:** ran pytest.\n```\n{output}\n```"]}

def critic_node(state: AgentState) -> dict:
    try:
        ast.parse(state["code"])
    except SyntaxError as exc:
        v = "retry" if state["iteration"] < 3 else "escalate"
        return {"feedback": str(exc), "verdict": v,
                "log": [f"🔍 **Critic:** syntax error — {exc}. Verdict: **{v.upper()}**."]}
    results = state["results"]
    has_fail = "failed" in results.lower() or "error" in results.lower()
    if not has_fail and "passed" in results.lower():
        return {"feedback": "All tests passed!", "verdict": "pass",
                "log": ["✅ **Critic:** all tests passed. Verdict: **PASS**."]}
    failed = [l.strip() for l in results.splitlines() if "FAILED" in l]
    diag = "Failing: " + "; ".join(failed) if failed else "Some tests failed."
    if state["iteration"] < 3:
        return {"feedback": diag, "verdict": "retry",
                "log": [f"🔄 **Critic:** {diag}\nVerdict: **RETRY**."]}
    return {"feedback": diag, "verdict": "escalate",
            "log": [f"🚨 **Critic:** {diag}\nVerdict: **ESCALATE** (max iterations)."]}

# ── LangGraph wiring ─────────────────────────────────────────────────────────
def _route(state: AgentState) -> str: return state["verdict"]

def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("planner", planner_node), ("developer", developer_node),
                     ("test_agent", test_agent_node), ("executor", executor_node),
                     ("critic", critic_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "developer")       # fan-out: parallel
    g.add_edge("planner", "test_agent")      # fan-out: parallel
    g.add_edge("developer", "executor")      # fan-in
    g.add_edge("test_agent", "executor")     # fan-in
    g.add_edge("executor", "critic")
    g.add_conditional_edges("critic", _route,
                            {"retry": "developer", "pass": END, "escalate": END})
    return g.compile()

# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AutoDev Agents", page_icon="🤖", layout="wide")
    st.title("🤖 AutoDev Agents")
    st.caption("Multi-agent pipeline: **Plan → Code → Test → Fix → Repeat** · LLM is mocked")
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected = st.selectbox("Coding requirement", list(PRESETS.keys()))
        run = st.button("🚀 Run Agents", type="primary", use_container_width=True)
        st.divider()
        st.caption("Planner → Developer ∥ Test Agent → Executor → Critic (retry loop)")

    if run:
        graph = build_graph()
        init = {"requirement": selected, "plan": "", "code": "", "tests": "",
                "results": "", "feedback": "", "iteration": 0, "verdict": "", "log": []}
        cv, rh, final = [], [], dict(init)

        with st.status("Running agent pipeline…", expanded=True) as status:
            try:
                for event in graph.stream(init, stream_mode="updates"):
                    for node, upd in event.items():
                        if node.startswith("__"): continue
                        for e in upd.get("log", []): status.markdown(e)
                        if "code" in upd: cv.append(upd["code"])
                        if "results" in upd: rh.append(upd["results"])
                        final.update({k: v for k, v in upd.items() if k != "log"})
                status.update(label="Pipeline finished!", state="complete")
            except Exception as exc:
                status.update(label=f"Pipeline error: {exc}", state="error")
                st.error(f"Unexpected error: {exc}"); return
        st.session_state["res"] = {"cv": cv, "rh": rh, "f": final}

    if "res" not in st.session_state:
        st.info("👈 Select a preset and press **Run Agents** to start."); return
    data = st.session_state["res"]
    f, cv, rh = data["f"], data["cv"], data["rh"]
    if f["verdict"] == "pass": st.success("🎉 All tests passed — pipeline succeeded!")
    elif f["verdict"] == "escalate": st.error("🚨 Escalated — manual intervention needed.")
    last = rh[-1] if rh else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", last)) else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Attempts", f["iteration"])
    c2.metric("Tests Passed", f"{passed}/5")
    c3.metric("Verdict", f["verdict"].upper())

    tab_code, tab_tests, tab_results = st.tabs(["💻 Code", "🧪 Tests", "📊 Results"])
    with tab_code:
        if len(cv) >= 2:
            left, right = st.columns(2)
            with left:
                st.subheader("v1 · initial (buggy)")
                st.code(cv[0], language="python")
            with right:
                st.subheader(f"v{f['iteration']} · final (fixed)")
                st.code(cv[-1], language="python")
        elif cv:
            st.code(cv[0], language="python")
    with tab_tests:
        st.code(f["tests"], language="python")
    with tab_results:
        for i, res in enumerate(rh, 1):
            with st.expander(f"Attempt {i}", expanded=(i == len(rh))):
                st.code(res)

if __name__ == "__main__":
    main()
