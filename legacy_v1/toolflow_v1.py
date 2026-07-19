"""ToolFlow: a controlled synthetic enterprise tool-use environment for studying
observation-conditioned plan repair in masked-diffusion planners.

Sequence layout (fixed length L=25):
  pos 0        <ctx>
  pos 1        task token            (TASK_REFUND | TASK_SUPPORT)
  pos 2        target customer name  (NAME_i)
  pos 3        distractor name       (NAME_j, j != i)
  pos 4..8     obs slot for steps 1..5 (init <none>)
  pos 9        <plan>
  pos 10..24   plan: 5 steps x 3 tokens (TOOL, ARG1, ARG2)

Two task families, both 5 steps:
  A REFUND : lookup_customer, check_account, get_order, refund|escalate, notify
             - perturbation: account SUSPENDED (revealed at obs2) -> step4 must
               become `escalate $1` (single-slot substitution repair)
             - intrinsic hole: refund ARG2 must be CODE_STD|CODE_PREMIUM matching
               the order tier revealed at obs3 (typed hole, fill-on-observation)
  B SUPPORT: lookup_customer, check_system, open_ticket|log_case, resolve,
             close_ticket|close_case
             - perturbation: ticket system down (revealed at obs2) -> steps 3 AND 5
               must change while step 4 (`resolve $3`) stays valid
               (non-contiguous dependent repair; preserves a valid middle step)

Success is state-diff based: goal effect set must be a subset of achieved effects.
"""
from dataclasses import dataclass, field
import random

# ----------------------------------------------------------------------------- vocab
SPECIALS = ["<pad>", "<mask>", "<ctx>", "<plan>", "<noarg>", "<none>"]
TASK_TOKENS = ["TASK_REFUND", "TASK_SUPPORT"]
NAMES = [f"NAME_{i}" for i in range(8)]
TOOLS = [
    "lookup_customer", "check_account", "get_order", "refund", "escalate",
    "notify", "check_system", "open_ticket", "log_case", "resolve",
    "close_ticket", "close_case", "noop",
]
VARS = [f"$%d" % i for i in range(1, 6)]           # $1..$5 : binding of step i
CODES = ["CODE_STD", "CODE_PREMIUM"]
OBS_TOKENS = [
    "OK", "ERR", "BOUND", "STATUS_ACTIVE", "STATUS_SUSPENDED",
    "ORDER_STD", "ORDER_PREMIUM", "SYS_OK", "ERR_TICKET_SYS",
]
VOCAB = SPECIALS + TASK_TOKENS + NAMES + TOOLS + VARS + CODES + OBS_TOKENS
TOK = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)

PAD, MASK, CTX, PLAN, NOARG, NONE = (TOK[t] for t in SPECIALS)
L = 25                      # total sequence length
OBS_POS = [4, 5, 6, 7, 8]   # obs slot for step i is OBS_POS[i-1]
PLAN_START = 10
N_STEPS, STEP_W = 5, 3
PLAN_POS = list(range(PLAN_START, PLAN_START + N_STEPS * STEP_W))

def step_slots(i):
    """Token positions of plan step i (1-indexed)."""
    s = PLAN_START + (i - 1) * STEP_W
    return [s, s + 1, s + 2]

# ------------------------------------------------------------------------- instances
@dataclass
class Instance:
    family: str            # 'A' (refund) or 'B' (support)
    name: str
    distractor: str
    perturbed: bool        # A: account suspended; B: ticket system down
    tier: str = "STD"      # A only: order tier -> required refund code

def sample_instance(rng: random.Random, p_perturb: float) -> Instance:
    fam = rng.choice(["A", "B"])
    name, distr = rng.sample(NAMES, 2)
    pert = rng.random() < p_perturb
    tier = rng.choice(["STD", "PREMIUM"])
    return Instance(fam, name, distr, pert, tier)

def initial_context(inst: Instance):
    seq = [PAD] * L
    seq[0] = CTX
    seq[1] = TOK["TASK_REFUND" if inst.family == "A" else "TASK_SUPPORT"]
    seq[2] = TOK[inst.name]
    seq[3] = TOK[inst.distractor]
    for p in OBS_POS:
        seq[p] = NONE
    seq[9] = PLAN
    for p in PLAN_POS:
        seq[p] = MASK
    return seq

# ------------------------------------------------------------------ expert plans
def _steps_to_tokens(steps):
    out = []
    for (tool, a1, a2) in steps:
        out += [TOK[tool], TOK[a1], TOK[a2]]
    return out

def expert_default_plan(inst: Instance):
    """Optimistic plan written before any observation. For family A the refund
    code (step4 ARG2) is supervised with the *true* code; since the tier is not
    observable at k=0, the learned marginal is ~uniform -> a natural typed hole."""
    if inst.family == "A":
        code = f"CODE_{inst.tier}"
        steps = [
            ("lookup_customer", inst.name, "<noarg>"),
            ("check_account", "$1", "<noarg>"),
            ("get_order", "$1", "<noarg>"),
            ("refund", "$3", code),
            ("notify", "$1", "<noarg>"),
        ]
    else:
        steps = [
            ("lookup_customer", inst.name, "<noarg>"),
            ("check_system", "<noarg>", "<noarg>"),
            ("open_ticket", "$1", "<noarg>"),
            ("resolve", "$3", "<noarg>"),
            ("close_ticket", "$3", "<noarg>"),
        ]
    return _steps_to_tokens(steps)

def expert_plan_given(inst: Instance, k: int):
    """Correct plan given observations revealed after executing steps 1..k."""
    plan = expert_default_plan(inst)
    rel = lambda i: step_slots(i)[0] - PLAN_START
    if inst.family == "A":
        if inst.perturbed and k >= 2:      # SUSPENDED known -> escalate
            plan[rel(4)] = TOK["escalate"]
            plan[rel(4) + 1] = TOK["$1"]
            plan[rel(4) + 2] = NOARG
        # tier known at k>=3 already matches default (true code supervised)
    else:
        if inst.perturbed and k >= 2:      # ticket system down
            plan[rel(3)] = TOK["log_case"]
            plan[rel(5)] = TOK["close_case"]
    return plan

def oracle_invalid_positions(inst: Instance, cur_plan_tokens, k: int):
    """Positions among *future* steps whose current token contradicts what is now
    known (diff against the expert plan under revealed information)."""
    target = expert_plan_given(inst, k)
    bad = []
    for i in range(k + 1, N_STEPS + 1):
        for p in step_slots(i):
            if cur_plan_tokens[p - PLAN_START] != target[p - PLAN_START]:
                bad.append(p)
    return bad

# ------------------------------------------------------------------------ execution
@dataclass
class EnvState:
    inst: Instance
    bindings: dict = field(default_factory=dict)   # step_idx -> kind
    effects: set = field(default_factory=set)
    invalid_calls: int = 0

def goal_effects(inst: Instance):
    if inst.family == "A":
        return {"escalated", "notified"} if inst.perturbed else {"refunded", "notified"}
    return {"resolved", "case_closed"} if inst.perturbed else {"resolved", "ticket_closed"}

def _resolve_arg(st: EnvState, tok_id):
    name = VOCAB[tok_id]
    if name.startswith("$"):
        return st.bindings.get(int(name[1:]))
    return name

def execute_step(st: EnvState, step_idx: int, tool_id, a1_id, a2_id):
    """Returns obs token id. Rejected calls increment invalid_calls and yield ERR."""
    inst, tool = st.inst, VOCAB[tool_id]
    a1 = _resolve_arg(st, a1_id)
    a2 = VOCAB[a2_id]
    def rej():
        st.invalid_calls += 1
        return TOK["ERR"]
    if tool == "lookup_customer":
        if a1 == inst.name:
            st.bindings[step_idx] = "cust"; return TOK["BOUND"]
        return rej()
    if tool == "check_account":
        if a1 == "cust":
            return TOK["STATUS_SUSPENDED" if (inst.family == "A" and inst.perturbed)
                       else "STATUS_ACTIVE"]
        return rej()
    if tool == "get_order":
        if a1 == "cust":
            st.bindings[step_idx] = "order"
            return TOK["ORDER_PREMIUM" if inst.tier == "PREMIUM" else "ORDER_STD"]
        return rej()
    if tool == "refund":
        active = not (inst.family == "A" and inst.perturbed)
        if a1 == "order" and active and a2 == f"CODE_{inst.tier}":
            st.effects.add("refunded"); return TOK["OK"]
        return rej()
    if tool == "escalate":
        if a1 == "cust":
            st.effects.add("escalated"); return TOK["OK"]
        return rej()
    if tool == "notify":
        if a1 == "cust":
            st.effects.add("notified"); return TOK["OK"]
        return rej()
    if tool == "check_system":
        return TOK["ERR_TICKET_SYS" if (inst.family == "B" and inst.perturbed)
                   else "SYS_OK"]
    if tool == "open_ticket":
        if a1 == "cust" and not (inst.family == "B" and inst.perturbed):
            st.bindings[step_idx] = "ticket"; st.effects.add("ticket_opened")
            return TOK["BOUND"]
        return rej()
    if tool == "log_case":
        if a1 == "cust":
            st.bindings[step_idx] = "case"; st.effects.add("case_opened")
            return TOK["BOUND"]
        return rej()
    if tool == "resolve":
        if a1 in ("ticket", "case"):
            st.effects.add("resolved"); return TOK["OK"]
        return rej()
    if tool == "close_ticket":
        if a1 == "ticket":
            st.effects.add("ticket_closed"); return TOK["OK"]
        return rej()
    if tool == "close_case":
        if a1 == "case":
            st.effects.add("case_closed"); return TOK["OK"]
        return rej()
    if tool == "noop":
        return TOK["OK"]
    return rej()

def success(st: EnvState) -> bool:
    return goal_effects(st.inst) <= st.effects
