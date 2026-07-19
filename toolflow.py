"""ToolFlow-v2: harder synthetic tool-use environment for selective plan repair.

Sequence layout (L=38):
  0 <ctx> | 1 task | 2 customer name | 3 contact name | 4 distractor name
  5..12 obs slots for steps 1..8 (init <none>) | 13 <plan> | 14..37 plan (8x3)

Three 8-step families, five perturbation kinds, reveal steps in ():
  A REFUND : lookup, check_account(2), get_order(3:tier), get_invoice(4),
             lookup_contact, refund|escalate|apply_credit, archive, notify
             A1 suspended(2) -> s6 escalate $1 ; A2 invoice-open(4) -> s6
             apply_credit $1 ; A1+A2 -> escalate (dominance rule).
             Typed hole: refund CODE must match tier revealed at obs3.
  B SUPPORT: lookup, check_system(2), open_ticket|log_case, get_order, resolve,
             archive, close_ticket|close_case, notify
             B1 system-down(2) -> s3 log_case, s7 close_case (steps 4-6 preserved).
  C RMA    : lookup, check_warranty(2), create_rma|quote_repair, get_order(4:tier),
             ship_label|swap_device, resolve, archive, notify
             C1 out-of-warranty(2) -> s3 quote_repair ; C2 premium(4) -> s5
             swap_device. C1+C2 compose independently (step 4 preserved between).

Doubles (A1+A2, C1+C2) NEVER appear in training (n_perturb<=1); evaluating at
n_perturb=2 is therefore compositional generalization by construction.
Binding confusability: two same-type person bindings ($1 customer, $5 contact in A)
and three names in context make argument errors possible under resampling.
"""
from dataclasses import dataclass, field
import random

SPECIALS = ["<pad>", "<mask>", "<ctx>", "<plan>", "<noarg>", "<none>"]
TASK_TOKENS = ["TASK_REFUND", "TASK_SUPPORT", "TASK_RMA"]
NAMES = [f"NAME_{i}" for i in range(10)]
TOOLS = ["lookup_customer", "lookup_contact", "check_account", "get_order",
         "get_invoice", "refund", "escalate", "apply_credit", "notify",
         "check_system", "open_ticket", "log_case", "resolve", "close_ticket",
         "close_case", "check_warranty", "create_rma", "quote_repair",
         "ship_label", "swap_device", "archive", "noop"]
VARS = [f"$%d" % i for i in range(1, 9)]
CODES = ["CODE_STD", "CODE_PREMIUM"]
OBS_TOKENS = ["OK", "ERR", "BOUND", "STATUS_ACTIVE", "STATUS_SUSPENDED",
              "ORDER_STD", "ORDER_PREMIUM", "SYS_OK", "ERR_TICKET_SYS",
              "WARRANTY_IN", "WARRANTY_OUT", "INV_OPEN", "INV_PAID"]
VOCAB = SPECIALS + TASK_TOKENS + NAMES + TOOLS + VARS + CODES + OBS_TOKENS
TOK = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)

PAD, MASK, CTX, PLAN, NOARG, NONE = (TOK[t] for t in SPECIALS)
L = 38
OBS_POS = list(range(5, 13))
PLAN_START = 14
N_STEPS, STEP_W = 8, 3
PLAN_POS = list(range(PLAN_START, PLAN_START + N_STEPS * STEP_W))

def step_slots(i):
    s = PLAN_START + (i - 1) * STEP_W
    return [s, s + 1, s + 2]

# ------------------------------------------------------------------- instances
PERTURBS = {"A": ["A1", "A2"], "B": ["B1"], "C": ["C1", "C2"]}
REVEAL = {"A1": 2, "A2": 4, "B1": 2, "C1": 2, "C2": 4}
REPAIR_STEPS = {"A1": [6], "A2": [6], "B1": [3, 7], "C1": [3], "C2": [5]}

def superset_positions(inst, k):
    """At a reveal turn, the union of ALL revealed perturbations' repair steps
    (future only) -- including steps whose current tokens are already correct.
    Unlike the oracle diff, this does NOT encode which resolution is right."""
    if not any(REVEAL[p] == k for p in inst.perturbs):
        return []
    pos = []
    for p in inst.perturbs:
        if k >= REVEAL[p]:
            for st in REPAIR_STEPS[p]:
                if st > k:
                    pos += step_slots(st)
    return sorted(set(pos))

@dataclass
class Instance:
    family: str
    name: str
    contact: str
    distractor: str
    perturbs: frozenset       # subset of PERTURBS[family]
    tier: str = "STD"         # A only: independent 50/50 (the CODE hole)

def sample_instance(rng: random.Random, n_perturb) -> Instance:
    """n_perturb: int 0/1/2, or a float in [0,1] interpreted as P(one perturb)."""
    if isinstance(n_perturb, float):
        n = 1 if rng.random() < n_perturb else 0
    else:
        n = n_perturb
    fam = rng.choice(["A", "C"]) if n == 2 else rng.choice(["A", "B", "C"])
    ps = PERTURBS[fam]
    perturbs = frozenset(rng.sample(ps, n) if n <= len(ps) else ps)
    name, contact, distractor = rng.sample(NAMES, 3)
    tier = rng.choice(["STD", "PREMIUM"])
    return Instance(fam, name, contact, distractor, perturbs, tier)

def initial_context(inst: Instance):
    seq = [PAD] * L
    seq[0] = CTX
    seq[1] = TOK[{"A": "TASK_REFUND", "B": "TASK_SUPPORT", "C": "TASK_RMA"}[inst.family]]
    seq[2] = TOK[inst.name]
    seq[3] = TOK[inst.contact]
    seq[4] = TOK[inst.distractor]
    for p in OBS_POS:
        seq[p] = NONE
    seq[13] = PLAN
    for p in PLAN_POS:
        seq[p] = MASK
    return seq

# ---------------------------------------------------------------- expert plans
def _tok3(tool, a1, a2):
    return [TOK[tool], TOK[a1], TOK[a2]]

def expert_default_plan(inst: Instance):
    if inst.family == "A":
        code = f"CODE_{inst.tier}"
        steps = [("lookup_customer", inst.name, "<noarg>"),
                 ("check_account", "$1", "<noarg>"),
                 ("get_order", "$1", "<noarg>"),
                 ("get_invoice", "$3", "<noarg>"),
                 ("lookup_contact", inst.contact, "<noarg>"),
                 ("refund", "$3", code),
                 ("archive", "$3", "<noarg>"),
                 ("notify", "$1", "<noarg>")]
    elif inst.family == "B":
        steps = [("lookup_customer", inst.name, "<noarg>"),
                 ("check_system", "<noarg>", "<noarg>"),
                 ("open_ticket", "$1", "<noarg>"),
                 ("get_order", "$1", "<noarg>"),
                 ("resolve", "$3", "<noarg>"),
                 ("archive", "$4", "<noarg>"),
                 ("close_ticket", "$3", "<noarg>"),
                 ("notify", "$1", "<noarg>")]
    else:
        steps = [("lookup_customer", inst.name, "<noarg>"),
                 ("check_warranty", "$1", "<noarg>"),
                 ("create_rma", "$1", "<noarg>"),
                 ("get_order", "$1", "<noarg>"),
                 ("ship_label", "$3", "<noarg>"),
                 ("resolve", "$3", "<noarg>"),
                 ("archive", "$4", "<noarg>"),
                 ("notify", "$1", "<noarg>")]
    out = []
    for s in steps:
        out += _tok3(*s)
    return out

def _set_step(plan, i, tool, a1, a2):
    r = (i - 1) * STEP_W
    plan[r], plan[r + 1], plan[r + 2] = TOK[tool], TOK[a1], TOK[a2]

def expert_plan_given(inst: Instance, k: int):
    """Correct plan under observations revealed after executing steps 1..k."""
    plan = expert_default_plan(inst)
    rev = {p for p in inst.perturbs if k >= REVEAL[p]}
    if inst.family == "A":
        if "A1" in rev:
            _set_step(plan, 6, "escalate", "$1", "<noarg>")
        elif "A2" in rev:
            _set_step(plan, 6, "apply_credit", "$1", "<noarg>")
    elif inst.family == "B":
        if "B1" in rev:
            _set_step(plan, 3, "log_case", "$1", "<noarg>")
            _set_step(plan, 7, "close_case", "$3", "<noarg>")
    else:
        if "C1" in rev:
            _set_step(plan, 3, "quote_repair", "$1", "<noarg>")
        if "C2" in rev:
            _set_step(plan, 5, "swap_device", "$3", "<noarg>")
    return plan

def oracle_invalid_positions(inst: Instance, cur_plan_tokens, k: int):
    target = expert_plan_given(inst, k)
    bad = []
    for i in range(k + 1, N_STEPS + 1):
        for p in step_slots(i):
            if cur_plan_tokens[p - PLAN_START] != target[p - PLAN_START]:
                bad.append(p)
    return bad

# ------------------------------------------------------------------- execution
@dataclass
class EnvState:
    inst: Instance
    bindings: dict = field(default_factory=dict)
    effects: set = field(default_factory=set)
    invalid_calls: int = 0

def goal_effects(inst: Instance):
    P = inst.perturbs
    if inst.family == "A":
        pay = "escalated" if "A1" in P else ("credited" if "A2" in P else "refunded")
        return {pay, "archived", "notified"}
    if inst.family == "B":
        return {"resolved", "archived",
                "case_closed" if "B1" in P else "ticket_closed", "notified"}
    return {"swapped" if "C2" in P else "shipped",
            "resolved", "archived", "notified"}

def _arg(st, tok_id):
    name = VOCAB[tok_id]
    if name.startswith("$"):
        return st.bindings.get(int(name[1:]))
    return name

def execute_step(st: EnvState, step_idx: int, tool_id, a1_id, a2_id):
    inst, tool = st.inst, VOCAB[tool_id]
    P = inst.perturbs
    a1 = _arg(st, a1_id)
    a2 = VOCAB[a2_id]
    def rej():
        st.invalid_calls += 1
        return TOK["ERR"]
    if tool == "lookup_customer":
        if a1 == inst.name:
            st.bindings[step_idx] = "cust"; return TOK["BOUND"]
        return rej()
    if tool == "lookup_contact":
        if a1 == inst.contact:
            st.bindings[step_idx] = "contact"; return TOK["BOUND"]
        return rej()
    if tool == "check_account":
        if a1 == "cust":
            return TOK["STATUS_SUSPENDED" if "A1" in P else "STATUS_ACTIVE"]
        return rej()
    if tool == "get_order":
        if a1 == "cust":
            st.bindings[step_idx] = "order"
            if inst.family == "A":
                return TOK["ORDER_PREMIUM" if inst.tier == "PREMIUM" else "ORDER_STD"]
            if inst.family == "C":
                return TOK["ORDER_PREMIUM" if "C2" in P else "ORDER_STD"]
            return TOK["BOUND"]
        return rej()
    if tool == "get_invoice":
        if a1 == "order":
            st.bindings[step_idx] = "invoice"
            return TOK["INV_OPEN" if "A2" in P else "INV_PAID"]
        return rej()
    if tool == "refund":
        ok = (a1 == "order" and "A1" not in P and "A2" not in P
              and a2 == f"CODE_{inst.tier}")
        if ok:
            st.effects.add("refunded"); return TOK["OK"]
        return rej()
    if tool == "escalate":
        if a1 == "cust":
            st.effects.add("escalated"); return TOK["OK"]
        return rej()
    if tool == "apply_credit":
        if a1 == "cust" and "A1" not in P:
            st.effects.add("credited"); return TOK["OK"]
        return rej()
    if tool == "notify":
        if a1 == "cust":
            st.effects.add("notified"); return TOK["OK"]
        return rej()
    if tool == "check_system":
        return TOK["ERR_TICKET_SYS" if "B1" in P else "SYS_OK"]
    if tool == "open_ticket":
        if a1 == "cust" and "B1" not in P:
            st.bindings[step_idx] = "ticket"; return TOK["BOUND"]
        return rej()
    if tool == "log_case":
        if a1 == "cust":
            st.bindings[step_idx] = "case"; return TOK["BOUND"]
        return rej()
    if tool == "resolve":
        if a1 in ("ticket", "case", "rma", "quote"):
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
    if tool == "check_warranty":
        if a1 == "cust":
            return TOK["WARRANTY_OUT" if "C1" in P else "WARRANTY_IN"]
        return rej()
    if tool == "create_rma":
        if a1 == "cust" and "C1" not in P:
            st.bindings[step_idx] = "rma"; return TOK["BOUND"]
        return rej()
    if tool == "quote_repair":
        if a1 == "cust":
            st.bindings[step_idx] = "quote"; return TOK["BOUND"]
        return rej()
    if tool == "ship_label":
        if a1 in ("rma", "quote") and not (inst.family == "C" and "C2" in P):
            st.effects.add("shipped"); return TOK["OK"]
        return rej()
    if tool == "swap_device":
        if a1 in ("rma", "quote"):
            st.effects.add("swapped"); return TOK["OK"]
        return rej()
    if tool == "archive":
        if a1 in ("order", "invoice"):
            st.effects.add("archived"); return TOK["OK"]
        return rej()
    if tool == "noop":
        return TOK["OK"]
    return rej()

def success(st: EnvState) -> bool:
    return goal_effects(st.inst) <= st.effects
