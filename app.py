import os
import sqlite3
from datetime import datetime
from typing import Set, Optional, Dict, Any, List

import pandas as pd
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Account Simulator", layout="wide")
DB_PATH = "account_game.db"

# =========================================================
# DB UTILS
# =========================================================
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def get_table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return set(r[1] for r in rows)

def safe_add_column(conn: sqlite3.Connection, table: str, coldef: str):
    col = coldef.split()[0]
    if col not in get_table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef};")

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")

# =========================================================
# INIT / MIGRATIONS
# =========================================================
def init_db_fresh():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rules(
      id INTEGER PRIMARY KEY CHECK(id=1),
      account_cost_usd  REAL    NOT NULL DEFAULT 216,
      reset_cost_usd    REAL    NOT NULL DEFAULT 100,
      max_resets        INTEGER NOT NULL DEFAULT 5,
      win_amount_usd    REAL    NOT NULL DEFAULT 4500,
      pro_threshold_usd REAL    NOT NULL DEFAULT 9000,
      cushion_usd       REAL    NOT NULL DEFAULT 4500,
      max_pro_accounts  INTEGER NOT NULL DEFAULT 5
    );
    """)
    cur.execute("INSERT OR IGNORE INTO rules(id) VALUES(1);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS participants(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      code TEXT NOT NULL,

      phase TEXT NOT NULL DEFAULT 'Eval',        -- Eval / Pro
      stage TEXT NOT NULL DEFAULT 'Eval',        -- Eval / Eval-Inactive / Pro-Pending / Pro-0 / Pro-4.5k / Pro-9k+ / BLOWN
      active INTEGER NOT NULL DEFAULT 1,
      blown INTEGER NOT NULL DEFAULT 0,

      resets_used INTEGER NOT NULL DEFAULT 0,
      purchase_paid_usd REAL NOT NULL DEFAULT 0,
      resets_paid_usd REAL NOT NULL DEFAULT 0,

      balance_usd REAL NOT NULL DEFAULT 0,       -- simulated
      withdrawable_usd REAL NOT NULL DEFAULT 0,  -- simulated

      created_at TEXT NOT NULL,

      FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS withdrawals(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      account_id INTEGER NOT NULL,

      percent REAL NOT NULL,
      base_amount_usd REAL NOT NULL,
      amount_usd REAL NOT NULL,

      created_at TEXT NOT NULL,
      note TEXT NOT NULL DEFAULT '',

      action TEXT NOT NULL DEFAULT 'blow',  -- always blow in this ruleset

      FOREIGN KEY(participant_id) REFERENCES participants(id),
      FOREIGN KEY(account_id) REFERENCES accounts(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      a_account_id INTEGER NOT NULL,
      b_account_id INTEGER NOT NULL,

      winner_account_id INTEGER NULL,
      loser_account_id INTEGER NULL,

      created_at TEXT NOT NULL,
      resolved_at TEXT NULL,

      note TEXT NOT NULL DEFAULT '',

      winner_balance_after REAL NULL,
      loser_balance_after  REAL NULL,
      winner_stage_after   TEXT NULL,
      loser_stage_after    TEXT NULL
    );
    """)

    conn.commit()
    conn.close()

def init_db_with_migrations():
    conn = db()
    cur = conn.cursor()

    # rules
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rules(
      id INTEGER PRIMARY KEY CHECK(id=1),
      account_cost_usd  REAL    NOT NULL DEFAULT 216,
      reset_cost_usd    REAL    NOT NULL DEFAULT 100,
      max_resets        INTEGER NOT NULL DEFAULT 5,
      win_amount_usd    REAL    NOT NULL DEFAULT 4500
    );
    """)
    cur.execute("INSERT OR IGNORE INTO rules(id) VALUES(1);")

    safe_add_column(conn, "rules", "pro_threshold_usd REAL NOT NULL DEFAULT 9000")
    safe_add_column(conn, "rules", "cushion_usd REAL NOT NULL DEFAULT 4500")
    safe_add_column(conn, "rules", "max_pro_accounts INTEGER NOT NULL DEFAULT 5")

    # participants
    cur.execute("""
    CREATE TABLE IF NOT EXISTS participants(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );
    """)

    # accounts base
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      code TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      resets_used INTEGER NOT NULL DEFAULT 0,
      purchase_paid_usd REAL NOT NULL DEFAULT 0,
      resets_paid_usd REAL NOT NULL DEFAULT 0,
      balance_usd REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)
    safe_add_column(conn, "accounts", "phase TEXT NOT NULL DEFAULT 'Eval'")
    safe_add_column(conn, "accounts", "stage TEXT NOT NULL DEFAULT 'Eval'")
    safe_add_column(conn, "accounts", "blown INTEGER NOT NULL DEFAULT 0")
    safe_add_column(conn, "accounts", "withdrawable_usd REAL NOT NULL DEFAULT 0")

    # withdrawals
    cur.execute("""
    CREATE TABLE IF NOT EXISTS withdrawals(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      account_id INTEGER NOT NULL,
      percent REAL NOT NULL,
      base_amount_usd REAL NOT NULL,
      amount_usd REAL NOT NULL,
      created_at TEXT NOT NULL,
      note TEXT NOT NULL DEFAULT '',
      FOREIGN KEY(participant_id) REFERENCES participants(id),
      FOREIGN KEY(account_id) REFERENCES accounts(id)
    );
    """)
    safe_add_column(conn, "withdrawals", "action TEXT NOT NULL DEFAULT 'blow'")

    # matches
    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      a_account_id INTEGER NOT NULL,
      b_account_id INTEGER NOT NULL,
      winner_account_id INTEGER NULL,
      loser_account_id INTEGER NULL,
      created_at TEXT NOT NULL,
      resolved_at TEXT NULL,
      note TEXT NOT NULL DEFAULT ''
    );
    """)
    safe_add_column(conn, "matches", "winner_balance_after REAL NULL")
    safe_add_column(conn, "matches", "loser_balance_after  REAL NULL")
    safe_add_column(conn, "matches", "winner_stage_after   TEXT NULL")
    safe_add_column(conn, "matches", "loser_stage_after    TEXT NULL")

    # rename legacy
    try:
        conn.execute("UPDATE accounts SET stage='Pro-Pending' WHERE stage='Qualified';")
    except Exception:
        pass

    conn.commit()
    conn.close()

# =========================================================
# RULES
# =========================================================
def get_rules() -> Dict[str, Any]:
    conn = db()
    row = conn.execute("""
      SELECT account_cost_usd, reset_cost_usd, max_resets, win_amount_usd,
             pro_threshold_usd, cushion_usd, max_pro_accounts
      FROM rules WHERE id=1
    """).fetchone()
    conn.close()
    return {
        "account_cost_usd": float(row[0]),
        "reset_cost_usd": float(row[1]),
        "max_resets": int(row[2]),
        "win_amount_usd": float(row[3]),
        "pro_threshold_usd": float(row[4]),
        "cushion_usd": float(row[5]),
        "max_pro_accounts": int(row[6]),
    }

def set_rules(r: Dict[str, Any]):
    conn = db()
    conn.execute("""
      UPDATE rules SET
        account_cost_usd=?,
        reset_cost_usd=?,
        max_resets=?,
        win_amount_usd=?,
        pro_threshold_usd=?,
        cushion_usd=?,
        max_pro_accounts=?
      WHERE id=1
    """, (
        float(r["account_cost_usd"]),
        float(r["reset_cost_usd"]),
        int(r["max_resets"]),
        float(r["win_amount_usd"]),
        float(r["pro_threshold_usd"]),
        float(r["cushion_usd"]),
        int(r["max_pro_accounts"]),
    ))
    conn.commit()
    conn.close()

# =========================================================
# PARTICIPANTS
# =========================================================
def list_participants():
    conn = db()
    df = pd.read_sql_query("SELECT id, name, created_at FROM participants ORDER BY name", conn)
    conn.close()
    return df

def add_participant(name: str):
    name = (name or "").strip()
    if not name:
        return
    conn = db()
    conn.execute("INSERT OR IGNORE INTO participants(name, created_at) VALUES(?, ?)", (name, now_iso()))
    conn.commit()
    conn.close()

def delete_participant(pid: int):
    conn = db()
    cur = conn.cursor()

    acc_ids = [r[0] for r in cur.execute("SELECT id FROM accounts WHERE participant_id=?", (pid,)).fetchall()]
    if acc_ids:
        q = ",".join(["?"] * len(acc_ids))
        cur.execute(f"DELETE FROM matches WHERE a_account_id IN ({q}) OR b_account_id IN ({q})", acc_ids + acc_ids)
        cur.execute(f"DELETE FROM withdrawals WHERE account_id IN ({q})", acc_ids)
        cur.execute("DELETE FROM accounts WHERE participant_id=?", (pid,))

    cur.execute("DELETE FROM withdrawals WHERE participant_id=?", (pid,))
    cur.execute("DELETE FROM participants WHERE id=?", (pid,))
    conn.commit()
    conn.close()

# =========================================================
# ACCOUNTS
# =========================================================
def list_accounts_full():
    conn = db()
    df = pd.read_sql_query("""
      SELECT a.id,
             p.name AS participant,
             p.id   AS participant_id,
             a.code,
             a.phase,
             a.stage,
             a.active,
             a.blown,
             a.resets_used,
             a.purchase_paid_usd,
             a.resets_paid_usd,
             a.balance_usd,
             a.withdrawable_usd,
             a.created_at
      FROM accounts a
      JOIN participants p ON p.id=a.participant_id
      ORDER BY p.name, a.id DESC
    """, conn)
    conn.close()
    return df

def count_pro_active() -> int:
    conn = db()
    n = conn.execute("SELECT COUNT(*) FROM accounts WHERE phase='Pro' AND blown=0").fetchone()[0]
    conn.close()
    return int(n)

def pro_slots_full(rules: Dict[str, Any]) -> bool:
    return count_pro_active() >= int(rules["max_pro_accounts"])

def eval_lives_left(resets_used: int, rules: Dict[str, Any]) -> int:
    # 1 initial + max_resets
    return max(0, 1 + int(rules["max_resets"]) - int(resets_used))

def recompute_account_state(account_id: int, rules: Dict[str, Any]):
    """
    Single source of truth:
    - auto promote Eval -> Pro (if threshold reached and slot available)
    - withdrawable calc
    - stage calc (non-sticky)
    """
    conn = db()
    cur = conn.cursor()
    row = cur.execute("""
        SELECT phase, blown, active, balance_usd
        FROM accounts
        WHERE id=?
    """, (account_id,)).fetchone()
    if not row:
        conn.close()
        return

    phase, blown, active, bal = row[0], int(row[1]), int(row[2]), float(row[3])

    # Auto promote Eval -> Pro (consumes eval profit => Pro balance 0)
    if blown == 0 and phase == "Eval" and active == 1 and bal >= float(rules["pro_threshold_usd"]):
        if not pro_slots_full(rules):
            phase = "Pro"
            bal = 0.0
            cur.execute("""
                UPDATE accounts
                SET phase='Pro', balance_usd=0, withdrawable_usd=0
                WHERE id=?
            """, (account_id,))
        # else stays Eval and becomes Pro-Pending via stage calc

    # Withdrawable only for Pro (simulated)
    wd = 0.0
    if blown == 0 and phase == "Pro":
        wd = max(0.0, bal - float(rules["cushion_usd"]))
    cur.execute("UPDATE accounts SET withdrawable_usd=? WHERE id=?", (wd, account_id))

    # Stage
    if blown == 1:
        stage = "BLOWN"
    elif phase == "Eval":
        if active == 0:
            stage = "Eval-Inactive"
        else:
            if bal >= float(rules["pro_threshold_usd"]) and pro_slots_full(rules):
                stage = "Pro-Pending"
            else:
                stage = "Eval"
    else:
        # Pro buckets (still useful for UI tier display)
        if bal < float(rules["cushion_usd"]):
            stage = "Pro-0"
        elif bal < 2 * float(rules["cushion_usd"]):
            stage = "Pro-4.5k"
        else:
            stage = "Pro-9k+"

    cur.execute("UPDATE accounts SET stage=? WHERE id=?", (stage, account_id))
    conn.commit()
    conn.close()

def next_account_code(pid: int, pname: str) -> str:
    prefix = "".join([c for c in (pname or "").upper() if c.isalnum()])[:3] or f"P{pid}"
    conn = db()
    n = conn.execute("SELECT COUNT(*) FROM accounts WHERE participant_id=?", (pid,)).fetchone()[0]
    conn.close()
    return f"{prefix}-{n+1:04d}"

def buy_account(pid: int, pname: str, rules: Dict[str, Any]):
    code = next_account_code(pid, pname)
    conn = db()
    conn.execute("""
      INSERT INTO accounts(participant_id, code, phase, stage, active, blown, resets_used,
                           purchase_paid_usd, resets_paid_usd, balance_usd, withdrawable_usd, created_at)
      VALUES(?, ?, 'Eval', 'Eval', 1, 0, 0, ?, 0, 0, 0, ?)
    """, (pid, code, float(rules["account_cost_usd"]), now_iso()))
    conn.commit()
    conn.close()

def manual_reset_eval(account_id: int, rules: Dict[str, Any]) -> str:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("""
        SELECT phase, blown, active, resets_used
        FROM accounts
        WHERE id=?
    """, (account_id,)).fetchone()
    if not row:
        conn.close()
        return "Account not found."

    phase, blown, active, resets_used = row[0], int(row[1]), int(row[2]), int(row[3])

    if blown == 1:
        conn.close()
        return "Account is BLOWN and cannot be reset."
    if phase != "Eval":
        conn.close()
        return "Only Eval accounts can be reset."
    if active == 1:
        conn.close()
        return "Account is already active."
    if resets_used >= int(rules["max_resets"]):
        conn.close()
        return f"No resets left ({resets_used}/{rules['max_resets']})."

    resets_used += 1
    cur.execute("""
      UPDATE accounts
      SET resets_used=?,
          resets_paid_usd = resets_paid_usd + ?,
          active=1,
          balance_usd=0
      WHERE id=?
    """, (resets_used, float(rules["reset_cost_usd"]), account_id))
    conn.commit()
    conn.close()

    recompute_account_state(account_id, rules)
    return f"Reset done. Used {resets_used}/{rules['max_resets']}."

def delete_account(account_id: int):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM matches WHERE a_account_id=? OR b_account_id=?", (account_id, account_id))
    cur.execute("DELETE FROM withdrawals WHERE account_id=?", (account_id,))
    cur.execute("DELETE FROM accounts WHERE id=?", (account_id,))
    conn.commit()
    conn.close()

# =========================================================
# WITHDRAWALS (ALWAYS BLOW)
# =========================================================
def withdrawals_df():
    conn = db()
    df = pd.read_sql_query("""
      SELECT w.id,
             p.name AS participant,
             w.participant_id,
             w.account_id,
             w.percent,
             w.base_amount_usd,
             w.amount_usd,
             w.created_at,
             w.note,
             w.action
      FROM withdrawals w
      JOIN participants p ON p.id=w.participant_id
      ORDER BY w.id DESC
    """, conn)
    conn.close()
    return df

def pro_withdraw_tier(balance_usd: float, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cushion = float(rules["cushion_usd"])
    if balance_usd < cushion:
        return None
    if balance_usd < 2 * cushion:
        return {"pct": 0.50, "name": "50% (Pro-4.5k)", "action": "blow"}
    return {"pct": 0.80, "name": "80% (Pro-9k+)", "action": "blow"}

def record_withdrawal(participant_id: int, account_id: int, note: str, rules: Dict[str, Any]) -> str:
    conn = db()
    cur = conn.cursor()

    row = cur.execute("""
        SELECT phase, blown, balance_usd
        FROM accounts WHERE id=?
    """, (account_id,)).fetchone()
    if not row:
        conn.close()
        return "Account not found."

    phase, blown, bal = row[0], int(row[1]), float(row[2])
    if blown == 1:
        conn.close()
        return "Account is BLOWN."
    if phase != "Pro":
        conn.close()
        return "Only Pro accounts can withdraw."

    tier = pro_withdraw_tier(bal, rules)
    if tier is None:
        conn.close()
        return "This Pro account is below cushion; withdrawals not allowed."

    base = float(rules["cushion_usd"])
    pct = float(tier["pct"])
    amount = pct * base

    cur.execute("""
      INSERT INTO withdrawals(participant_id, account_id, percent, base_amount_usd, amount_usd, created_at, note, action)
      VALUES(?, ?, ?, ?, ?, ?, ?, 'blow')
    """, (participant_id, account_id, pct, base, amount, now_iso(), (note or "").strip()))

    # Always blow after withdrawal (your rule)
    cur.execute("""
      UPDATE accounts
      SET active=0, blown=1
      WHERE id=?
    """, (account_id,))

    conn.commit()
    conn.close()

    recompute_account_state(account_id, rules)
    return f"Withdrawal recorded: {amount:,.2f} USD | {tier['name']} | action=BLOW"

# =========================================================
# MATCH ENGINE + TIMELINE SNAPSHOTS
# =========================================================
def create_match(a_id: int, b_id: int, note: str = "") -> str:
    if a_id == b_id:
        return "Pick two different accounts."

    conn = db()
    rowA = conn.execute("SELECT active, blown FROM accounts WHERE id=?", (a_id,)).fetchone()
    rowB = conn.execute("SELECT active, blown FROM accounts WHERE id=?", (b_id,)).fetchone()
    if not rowA or not rowB:
        conn.close()
        return "Account not found."
    if int(rowA[0]) != 1 or int(rowB[0]) != 1 or int(rowA[1]) == 1 or int(rowB[1]) == 1:
        conn.close()
        return "Both accounts must be Active and not BLOWN."

    conn.execute("""
      INSERT INTO matches(a_account_id, b_account_id, created_at, note)
      VALUES(?, ?, ?, ?)
    """, (a_id, b_id, now_iso(), (note or "").strip()))
    conn.commit()
    conn.close()
    return "Match created."

def get_open_match() -> Optional[pd.Series]:
    conn = db()
    df = pd.read_sql_query("""
      SELECT id, a_account_id, b_account_id, created_at, note
      FROM matches
      WHERE resolved_at IS NULL
      ORDER BY id DESC
      LIMIT 1
    """, conn)
    conn.close()
    return None if df.empty else df.iloc[0]

def resolve_match(match_id: int, winner_id: int, loser_id: int, rules: Dict[str, Any]) -> str:
    win_amt = float(rules["win_amount_usd"])

    conn = db()
    cur = conn.cursor()

    # Winner always receives +win_amt
    cur.execute("UPDATE accounts SET balance_usd = balance_usd + ? WHERE id=?", (win_amt, winner_id))

    # Loser logic
    loser = cur.execute("SELECT phase FROM accounts WHERE id=?", (loser_id,)).fetchone()
    if not loser:
        conn.close()
        return "Loser account not found."
    loser_phase = loser[0]

    if loser_phase == "Eval":
        # Eval loses -> inactive (needs reset)
        cur.execute("""
          UPDATE accounts
          SET balance_usd = balance_usd - ?,
              active = 0
          WHERE id=?
        """, (win_amt, loser_id))
    else:
        # Pro loses -> ALWAYS BLOWN (your rule)
        cur.execute("""
          UPDATE accounts
          SET balance_usd = balance_usd - ?,
              active = 0,
              blown = 1
          WHERE id=?
        """, (win_amt, loser_id))

    # Close match
    cur.execute("""
      UPDATE matches
      SET winner_account_id=?,
          loser_account_id=?,
          resolved_at=?
      WHERE id=?
    """, (winner_id, loser_id, now_iso(), match_id))

    conn.commit()
    conn.close()

    # Recompute both accounts immediately
    recompute_account_state(winner_id, rules)
    recompute_account_state(loser_id, rules)

    # Store snapshots
    conn = db()
    cur = conn.cursor()
    w = cur.execute("SELECT balance_usd, stage FROM accounts WHERE id=?", (winner_id,)).fetchone()
    l = cur.execute("SELECT balance_usd, stage FROM accounts WHERE id=?", (loser_id,)).fetchone()
    cur.execute("""
      UPDATE matches
      SET winner_balance_after=?,
          loser_balance_after=?,
          winner_stage_after=?,
          loser_stage_after=?
      WHERE id=?
    """, (float(w[0]), float(l[0]), str(w[1]), str(l[1]), match_id))
    conn.commit()
    conn.close()

    return "Match resolved."

def matches_df(limit=200):
    conn = db()
    df = pd.read_sql_query(f"SELECT * FROM matches ORDER BY id DESC LIMIT {int(limit)}", conn)
    conn.close()
    return df

# =========================================================
# SUMMARY + STATUS MAP
# =========================================================
STATUS_META = {
    "Eval":          {"icon": "🟩", "label": "Eval (active)"},
    "Eval-Inactive": {"icon": "⬜", "label": "Eval inactive (needs reset)"},
    "Pro-Pending":   {"icon": "🟧", "label": "Hit 9k but waiting Pro slot"},
    "Pro-0":         {"icon": "🟨", "label": "Pro < 4.5k"},
    "Pro-4.5k":      {"icon": "🟦", "label": "Pro 4.5k–9k"},
    "Pro-9k+":       {"icon": "🟪", "label": "Pro 9k+"},
    "BLOWN":         {"icon": "🟥", "label": "BLOWN"},
}

def status_counts(df: pd.DataFrame) -> Dict[str, int]:
    keys = list(STATUS_META.keys())
    if df.empty:
        return {k: 0 for k in keys}
    vc = df["stage"].value_counts().to_dict()
    return {k: int(vc.get(k, 0)) for k in keys}

def global_summary(acc: pd.DataFrame, wd: pd.DataFrame):
    if acc.empty:
        return dict(accounts_total=0, spend=0.0, withdrawn=0.0, net=0.0)
    spend = float((acc["purchase_paid_usd"].astype(float) + acc["resets_paid_usd"].astype(float)).sum())
    withdrawn = float(wd["amount_usd"].sum()) if not wd.empty else 0.0
    net = withdrawn - spend
    return dict(accounts_total=len(acc), spend=spend, withdrawn=withdrawn, net=net)

# =========================================================
# MATCH INSIGHTS (neutral ranking)
# =========================================================
def loss_impact(row, rules: Dict[str, Any]) -> Dict[str, Any]:
    phase = str(row["phase"])
    resets_used = int(row["resets_used"])

    if phase == "Eval":
        lives = eval_lives_left(resets_used, rules)
        can_reset = resets_used < int(rules["max_resets"])
        return {
            "loser_state": "Eval-Inactive",
            "reset_cost": float(rules["reset_cost_usd"]) if can_reset else 0.0,
            "dead_if_loses": bool(not can_reset),
            "blow_risk": 0.0,
            "lives_left": lives,
        }

    # Pro loses always blown
    return {
        "loser_state": "BLOWN",
        "reset_cost": 0.0,
        "dead_if_loses": True,
        "blow_risk": 1.0,
        "lives_left": None,
    }

def winner_progress_risk(row, rules: Dict[str, Any]) -> Dict[str, Any]:
    phase = str(row["phase"])
    bal = float(row["balance_usd"])
    win_amt = float(rules["win_amount_usd"])
    threshold = float(rules["pro_threshold_usd"])

    if phase != "Eval":
        return {"would_hit_threshold": False, "would_be_pending": False}

    new_bal = bal + win_amt
    would_hit = new_bal >= threshold
    would_pending = would_hit and pro_slots_full(rules)
    return {"would_hit_threshold": bool(would_hit), "would_be_pending": bool(would_pending)}

def build_match_insights(all_accounts: pd.DataFrame, eligible: pd.DataFrame, rules: Dict[str, Any], prevent_same: bool) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame()

    idx = all_accounts.set_index("id")
    cand_ids = eligible["id"].tolist()

    rows: List[Dict[str, Any]] = []
    for i in range(len(cand_ids)):
        for j in range(i + 1, len(cand_ids)):
            a_id = int(cand_ids[i])
            b_id = int(cand_ids[j])
            a = idx.loc[a_id]
            b = idx.loc[b_id]
            if prevent_same and str(a["participant"]) == str(b["participant"]):
                continue

            a_loss = loss_impact(a, rules)
            b_loss = loss_impact(b, rules)
            a_win = winner_progress_risk(a, rules)
            b_win = winner_progress_risk(b, rules)

            reset_cost_exposure = a_loss["reset_cost"] + b_loss["reset_cost"]
            blow_risk_exposure = a_loss["blow_risk"] + b_loss["blow_risk"]
            pending_risk = (1 if a_win["would_be_pending"] else 0) + (1 if b_win["would_be_pending"] else 0)
            pro_progress = (1 if a_win["would_hit_threshold"] else 0) + (1 if b_win["would_hit_threshold"] else 0)

            rows.append({
                "A_participant": a["participant"],
                "A_code": a["code"],
                "A_stage": a["stage"],
                "B_participant": b["participant"],
                "B_code": b["code"],
                "B_stage": b["stage"],
                "reset_cost_exposure": float(reset_cost_exposure),
                "blow_risk_exposure": float(blow_risk_exposure),
                "pro_progress_signals": int(pro_progress),
                "pro_pending_signals": int(pending_risk),
            })

    return pd.DataFrame(rows)

def score_matchups(dfm: pd.DataFrame, objective: str) -> pd.DataFrame:
    if dfm.empty:
        return dfm
    if objective == "Minimize reset cost exposure":
        dfm["score"] = -(dfm["reset_cost_exposure"]) - 80 * dfm["blow_risk_exposure"] - 10 * dfm["pro_pending_signals"] + 6 * dfm["pro_progress_signals"]
    elif objective == "Minimize blow risk":
        dfm["score"] = -120 * dfm["blow_risk_exposure"] - 0.6 * dfm["reset_cost_exposure"] - 10 * dfm["pro_pending_signals"] + 6 * dfm["pro_progress_signals"]
    elif objective == "Maximize progress to Pro":
        dfm["score"] = 25 * dfm["pro_progress_signals"] - 1.2 * dfm["reset_cost_exposure"] - 80 * dfm["blow_risk_exposure"] - 10 * dfm["pro_pending_signals"]
    else:
        dfm["score"] = 12 * dfm["pro_progress_signals"] - 1.0 * dfm["reset_cost_exposure"] - 100 * dfm["blow_risk_exposure"] - 10 * dfm["pro_pending_signals"]
    return dfm.sort_values("score", ascending=False)

# =========================================================
# APP START
# =========================================================
init_db_with_migrations()
rules = get_rules()

st.title("Account Simulator — Ledger + Tournament")

# DB RESET
with st.expander("⚠️ Database reset", expanded=False):
    st.write("Deletes the SQLite file and recreates a clean database.")
    confirm = st.checkbox("I understand this deletes all data.", value=False)
    if st.button("RESET DATABASE NOW", type="primary", disabled=not confirm):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db_fresh()
        st.success("Database reset complete.")
        st.rerun()

# RULES
with st.expander("Global Rules", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7 = st.columns(3)

    r = dict(rules)
    r["account_cost_usd"] = c1.number_input("Account cost (USD)", min_value=0.0, value=r["account_cost_usd"], step=1.0)
    r["reset_cost_usd"] = c2.number_input("Reset cost (USD)", min_value=0.0, value=r["reset_cost_usd"], step=1.0)
    r["max_resets"] = c3.number_input("Max resets (Eval)", min_value=0, value=r["max_resets"], step=1)
    r["max_pro_accounts"] = c4.number_input("Max Pro accounts (global)", min_value=0, value=r["max_pro_accounts"], step=1)

    r["win_amount_usd"] = c5.number_input("Win/Loss amount (virtual)", min_value=0.0, value=r["win_amount_usd"], step=100.0)
    r["pro_threshold_usd"] = c6.number_input("Eval threshold to become Pro", min_value=0.0, value=r["pro_threshold_usd"], step=500.0)
    r["cushion_usd"] = c7.number_input("Cushion (Pro)", min_value=0.0, value=r["cushion_usd"], step=100.0)

    st.caption("Rules: Eval hits threshold => becomes Pro if slot exists, else Pro-Pending.")
    st.caption("Rules: Withdraw from Pro => ALWAYS BLOWN (50% if Pro-4.5k, 80% if Pro-9k+).")
    st.caption("Rules: Pro that loses a match => ALWAYS BLOWN.")
    if st.button("Save rules"):
        set_rules(r)
        st.success("Rules saved.")
        st.rerun()

participants = list_participants()
accounts = list_accounts_full()
withdrawals = withdrawals_df()
matches = matches_df(200)

# Compact status row (no legend)
counts = status_counts(accounts)
st.subheader("Status Map (compact)")
cols = st.columns(7)
keys = list(STATUS_META.keys())
for col, k in zip(cols, keys):
    meta = STATUS_META[k]
    col.metric(f"{meta['icon']} {k}", counts[k])

# GLOBAL SUMMARY
s = global_summary(accounts, withdrawals)
st.subheader("Global Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accounts", s["accounts_total"])
m2.metric("Spend (real)", f"{s['spend']:,.2f}")
m3.metric("Withdrawn (real)", f"{s['withdrawn']:,.2f}")
m4.metric("Net (real)", f"{s['net']:,.2f}")
m5.metric("Pro cap", f"{count_pro_active()}/{rules['max_pro_accounts']}")

st.divider()

# LEDGER
st.subheader("Ledger — Participants")
left, right = st.columns([1.1, 2.9], gap="large")

with left:
    st.markdown("### Add participant")
    new_name = st.text_input("Name", placeholder="Rene / Friend1 / Friend2")
    if st.button("Add participant", use_container_width=True):
        add_participant(new_name)
        st.rerun()

with right:
    if participants.empty:
        st.info("Add participants to start.")
    else:
        acc = accounts.copy()
        acc["spend_usd"] = acc["purchase_paid_usd"].astype(float) + acc["resets_paid_usd"].astype(float)
        wd_by = withdrawals.groupby("participant_id", as_index=False)["amount_usd"].sum().rename(columns={"amount_usd": "withdrawn_usd"}) if not withdrawals.empty else pd.DataFrame(columns=["participant_id", "withdrawn_usd"])

        byp = acc.groupby(["participant_id", "participant"], as_index=False).agg(
            accounts=("id", "count"),
            active=("active", "sum"),
            blown=("blown", "sum"),
            resets_used=("resets_used", "sum"),
            spend_usd=("spend_usd", "sum"),
        ).merge(wd_by, on="participant_id", how="left").fillna({"withdrawn_usd": 0.0})

        byp["net_usd"] = byp["withdrawn_usd"] - byp["spend_usd"]

        totals = {
            "participant_id": "",
            "participant": "TOTAL",
            "accounts": int(byp["accounts"].sum()),
            "active": int(byp["active"].sum()),
            "blown": int(byp["blown"].sum()),
            "resets_used": int(byp["resets_used"].sum()),
            "spend_usd": float(byp["spend_usd"].sum()),
            "withdrawn_usd": float(byp["withdrawn_usd"].sum()),
            "net_usd": float(byp["net_usd"].sum()),
        }
        byp_show = pd.concat([byp.sort_values("participant"), pd.DataFrame([totals])], ignore_index=True)

        st.markdown("### Per-participant summary (with TOTAL)")
        st.dataframe(
            byp_show[["participant", "accounts", "active", "blown", "resets_used", "spend_usd", "withdrawn_usd", "net_usd"]],
            use_container_width=True,
            hide_index=True
        )

# Keep expander open after actions
if "open_pid" not in st.session_state:
    st.session_state.open_pid = None

if not participants.empty:
    for _, p in participants.iterrows():
        pid = int(p["id"])
        pname = str(p["name"])

        p_acc = accounts[accounts["participant_id"] == pid].copy()
        p_wd = withdrawals[withdrawals["participant_id"] == pid].copy() if not withdrawals.empty else pd.DataFrame()

        p_spend = float((p_acc["purchase_paid_usd"].astype(float) + p_acc["resets_paid_usd"].astype(float)).sum()) if not p_acc.empty else 0.0
        p_withdrawn = float(p_wd["amount_usd"].sum()) if not p_wd.empty else 0.0
        p_net = p_withdrawn - p_spend

        expanded = (st.session_state.open_pid == pid)
        with st.expander(f"{pname} — Spend {p_spend:,.2f} | Withdrawn {p_withdrawn:,.2f} | Net {p_net:,.2f}", expanded=expanded):
            _, topR = st.columns([3, 1])
            with topR:
                if st.button("🗑️ Delete participant", key=f"del_{pid}"):
                    st.session_state.open_pid = None
                    delete_participant(pid)
                    st.rerun()

            cA, cB = st.columns([1.2, 2.8], gap="large")

            with cA:
                st.markdown("#### Buy")
                if st.button(f"Buy account (${rules['account_cost_usd']:.0f})", key=f"buy_{pid}", use_container_width=True):
                    st.session_state.open_pid = pid
                    buy_account(pid, pname, rules)
                    st.rerun()

                st.markdown("#### Manual reset (Eval-Inactive only)")
                reset_candidates = p_acc[
                    (p_acc["phase"] == "Eval") &
                    (p_acc["stage"] == "Eval-Inactive") &
                    (p_acc["blown"] == 0)
                ].copy()

                if reset_candidates.empty:
                    st.caption("No Eval-Inactive accounts to reset.")
                else:
                    reset_candidates["lives_left"] = reset_candidates["resets_used"].astype(int).apply(lambda x: eval_lives_left(x, rules))
                    reset_candidates["label"] = reset_candidates.apply(
                        lambda r: f"{r['code']} | resets {int(r['resets_used'])}/{rules['max_resets']} | lives left {int(r['lives_left'])}",
                        axis=1
                    )
                    sel = st.selectbox("Pick account", reset_candidates["label"].tolist(), key=f"reset_pick_{pid}")
                    acc_id = int(reset_candidates[reset_candidates["label"] == sel]["id"].iloc[0])
                    if st.button(f"Reset (+${rules['reset_cost_usd']:.0f})", key=f"reset_btn_{pid}", use_container_width=True):
                        st.session_state.open_pid = pid
                        msg = manual_reset_eval(acc_id, rules)
                        (st.success if msg.startswith("Reset") else st.warning)(msg)
                        st.rerun()

                st.markdown("#### Delete dead Eval accounts (no resets left)")
                dead = p_acc[
                    (p_acc["phase"] == "Eval") &
                    (p_acc["stage"] == "Eval-Inactive") &
                    (p_acc["resets_used"].astype(int) >= int(rules["max_resets"]))
                ].copy()

                if dead.empty:
                    st.caption("No dead Eval accounts.")
                else:
                    dead["label"] = dead.apply(lambda r: f"{r['code']} | resets {int(r['resets_used'])}/{rules['max_resets']}", axis=1)
                    sel_d = st.selectbox("Pick dead account", dead["label"].tolist(), key=f"del_acc_pick_{pid}")
                    del_id = int(dead[dead["label"] == sel_d]["id"].iloc[0])
                    if st.button("Delete account", key=f"del_acc_btn_{pid}", use_container_width=True):
                        st.session_state.open_pid = pid
                        delete_account(del_id)
                        st.rerun()

            with cB:
                with st.expander("Withdrawals (Pro only) — click to expand", expanded=False):
                    pro_acc = p_acc[(p_acc["phase"] == "Pro") & (p_acc["blown"] == 0)].copy()
                    if pro_acc.empty:
                        st.caption("No Pro accounts available.")
                    else:
                        pro_acc = pro_acc[pro_acc["balance_usd"].astype(float) >= float(rules["cushion_usd"])].copy()
                        if pro_acc.empty:
                            st.caption("No Pro accounts >= cushion (4.5k). Pro-0 is hidden here.")
                        else:
                            pro_acc["tier"] = pro_acc["balance_usd"].astype(float).apply(lambda b: pro_withdraw_tier(b, rules))
                            pro_acc["tier_name"] = pro_acc["tier"].apply(lambda t: t["name"] if t else "—")
                            pro_acc["label"] = pro_acc.apply(
                                lambda r: f"{r['code']} | {r['stage']} | {r['tier_name']} | bal {float(r['balance_usd']):,.0f}",
                                axis=1
                            )
                            selw = st.selectbox("Pick Pro", pro_acc["label"].tolist(), key=f"wd_pick_{pid}")
                            row_sel = pro_acc[pro_acc["label"] == selw].iloc[0]
                            acc_id = int(row_sel["id"])
                            bal = float(row_sel["balance_usd"])
                            tier = pro_withdraw_tier(bal, rules)

                            st.info(f"Auto tier: **{tier['name']}**")
                            st.warning("Action: **Withdrawal => Account becomes BLOWN** (always).")
                            note = st.text_input("Note", key=f"wd_note_{pid}")
                            if st.button("Record withdrawal", key=f"wd_btn_{pid}", use_container_width=True):
                                st.session_state.open_pid = pid
                                msg = record_withdrawal(pid, acc_id, note, rules)
                                (st.success if msg.startswith("Withdrawal") else st.warning)(msg)
                                st.rerun()

            st.markdown("#### Accounts")
            if p_acc.empty:
                st.write("No accounts yet.")
            else:
                view = p_acc.copy()
                view["spend_usd"] = view["purchase_paid_usd"].astype(float) + view["resets_paid_usd"].astype(float)
                view["lives_left"] = view.apply(
                    lambda r: eval_lives_left(int(r["resets_used"]), rules) if str(r["phase"]) == "Eval" else None,
                    axis=1
                )
                st.dataframe(
                    view[["code", "phase", "stage", "active", "blown", "resets_used", "lives_left", "balance_usd", "withdrawable_usd", "spend_usd"]],
                    use_container_width=True, hide_index=True
                )

            if not p_wd.empty:
                st.markdown("#### Withdrawal history")
                st.dataframe(
                    p_wd[["account_id", "percent", "base_amount_usd", "amount_usd", "action", "created_at", "note"]],
                    use_container_width=True, hide_index=True
                )

st.divider()

# =========================================================
# TOURNAMENT
# =========================================================
st.subheader("Tournament Engine — Manual Match Selection")

accounts = list_accounts_full()
eligible = accounts[(accounts["active"] == 1) & (accounts["blown"] == 0)].copy()

if eligible.empty:
    st.info("No eligible active accounts.")
else:
    prevent_same = st.checkbox("Prevent same participant fights", value=True)

    # IMPORTANT: remove Pro-Pending from tournament
    allowed_types = ["Eval", "Pro-0", "Pro-4.5k", "Pro-9k+"]
    types_sel = st.multiselect("Allowed types", allowed_types, default=allowed_types)
    eligible = eligible[eligible["stage"].isin(types_sel)].copy()

    with st.expander("Match Insights (simulator ranking)", expanded=False):
        objective = st.selectbox(
            "Objective",
            ["Balanced", "Minimize reset cost exposure", "Minimize blow risk", "Maximize progress to Pro"],
            index=0
        )
        dfm = build_match_insights(accounts, eligible, rules, prevent_same)
        dfm = score_matchups(dfm, objective)
        if dfm.empty:
            st.caption("No candidate matchups for current filters.")
        else:
            st.dataframe(dfm.head(15), use_container_width=True, hide_index=True)
            st.caption("Ranking is based on your chosen objective; it does not guarantee outcomes.")

    st.divider()

    part_list = sorted(eligible["participant"].unique().tolist())
    if not part_list:
        st.warning("No accounts match your type filter.")
    else:
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.8, 1.2, 1.8, 2.0], gap="large")

        with c1:
            pA = st.selectbox("Participant A", part_list, key="pA")
        poolA = eligible[eligible["participant"] == pA].copy()
        poolA["label"] = poolA.apply(lambda r: f"{r['code']} | {r['stage']} | bal {float(r['balance_usd']):,.0f}", axis=1)

        with c2:
            a_sel = st.selectbox("Account A", poolA["label"].tolist(), key="a_sel")
            a_id = int(poolA[poolA["label"] == a_sel]["id"].iloc[0])

        with c3:
            b_parts = [x for x in part_list if (not prevent_same or x != pA)]
            if not b_parts:
                b_parts = part_list
            pB = st.selectbox("Participant B", b_parts, key="pB")
        poolB = eligible[eligible["participant"] == pB].copy()
        poolB["label"] = poolB.apply(lambda r: f"{r['code']} | {r['stage']} | bal {float(r['balance_usd']):,.0f}", axis=1)

        with c4:
            b_sel = st.selectbox("Account B", poolB["label"].tolist(), key="b_sel")
            b_id = int(poolB[poolB["label"] == b_sel]["id"].iloc[0])

        with c5:
            note = st.text_input("Match note (optional)", key="match_note")

        st.markdown("#### Coach (impact preview)")
        idx = accounts.set_index("id")
        a = idx.loc[a_id]
        b = idx.loc[b_id]
        a_loss = loss_impact(a, rules)
        b_loss = loss_impact(b, rules)
        st.info(
            f"If **A loses**: {a_loss['loser_state']} | reset_cost={a_loss['reset_cost']:.0f} | blow_risk={a_loss['blow_risk']}\n\n"
            f"If **B loses**: {b_loss['loser_state']} | reset_cost={b_loss['reset_cost']:.0f} | blow_risk={b_loss['blow_risk']}"
        )

        if st.button("Create match", type="primary"):
            if prevent_same and pA == pB:
                st.warning("Same participant. Pick another opponent or disable the checkbox.")
            else:
                msg = create_match(a_id, b_id, note)
                (st.success if msg.startswith("Match created") else st.warning)(msg)
                st.rerun()

        open_m = get_open_match()
        if open_m is None:
            st.info("No open match.")
        else:
            idx2 = list_accounts_full().set_index("id")
            a = idx2.loc[int(open_m["a_account_id"])]
            b = idx2.loc[int(open_m["b_account_id"])]

            st.markdown("### Open match")
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.markdown(f"**A:** {a['participant']} — {a['code']}")
                st.write(f"{a['stage']} | Bal {float(a['balance_usd']):,.0f}")
                if st.button("✅ A WINS", type="primary", use_container_width=True):
                    msg = resolve_match(int(open_m["id"]), int(open_m["a_account_id"]), int(open_m["b_account_id"]), rules)
                    st.success(msg)
                    st.rerun()
            with cb:
                st.markdown(f"**B:** {b['participant']} — {b['code']}")
                st.write(f"{b['stage']} | Bal {float(b['balance_usd']):,.0f}")
                if st.button("✅ B WINS", type="primary", use_container_width=True):
                    msg = resolve_match(int(open_m["id"]), int(open_m["b_account_id"]), int(open_m["a_account_id"]), rules)
                    st.success(msg)
                    st.rerun()

st.divider()

# =========================================================
# TIMELINE (no duplicate A/B row)
# =========================================================
st.subheader("Match Timeline — Chronological")

m = matches_df(200)
accounts = list_accounts_full()

if m.empty:
    st.info("No matches yet.")
else:
    m_chrono = m.iloc[::-1].copy()
    show_n = st.slider("Show last N resolved matches", min_value=10, max_value=200, value=60, step=10)
    m_chrono = m_chrono[m_chrono["resolved_at"].notna()].tail(show_n)

    idx = accounts.set_index("id")

    def acc_short(aid):
        if pd.isna(aid):
            return "—"
        aid = int(aid)
        if aid not in idx.index:
            return f"#{aid}"
        r = idx.loc[aid]
        return f"{r['participant']} | {r['code']}"

    for _, row in m_chrono.iterrows():
        mid = int(row["id"])
        w_id = int(row["winner_account_id"]) if pd.notna(row["winner_account_id"]) else None
        l_id = int(row["loser_account_id"]) if pd.notna(row["loser_account_id"]) else None

        with st.container(border=True):
            top = st.columns([1.0, 2.0])
            top[0].markdown(f"**M#{mid}**")
            top[1].write(f"**Resolved:** {row['resolved_at']}")

            bot = st.columns([2.0, 2.0, 2.0, 2.0])
            bot[0].markdown(f"✅ **Winner:** {acc_short(w_id)}")
            bot[1].markdown(f"📌 **Winner after:** {row['winner_stage_after'] or '—'} | bal {float(row['winner_balance_after']) if pd.notna(row['winner_balance_after']) else 0:,.0f}")
            bot[2].markdown(f"❌ **Loser:** {acc_short(l_id)}")
            bot[3].markdown(f"📌 **Loser after:** {row['loser_stage_after'] or '—'} | bal {float(row['loser_balance_after']) if pd.notna(row['loser_balance_after']) else 0:,.0f}")

            if str(row.get("note") or "").strip():
                st.caption(f"Note: {row['note']}")

st.caption("Simulator: spend/withdrawals are game ledger entries; balances are simulated values.")
