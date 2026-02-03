import os
import sqlite3
from datetime import datetime
from typing import Set, Optional

import pandas as pd
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Account Stats Game", layout="wide")
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
# INIT (FRESH SCHEMA)
# =========================================================
def init_db_fresh():
    """Create fresh schema (assumes DB file is new/empty)."""
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

    # NOTE:
    # - balance_usd is "virtual game balance"
    # - spend is real money: purchase_paid_usd + resets_paid_usd
    # - withdrawals table is real money events you record manually
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      code TEXT NOT NULL,

      phase TEXT NOT NULL DEFAULT 'Eval',         -- Eval / Pro
      active INTEGER NOT NULL DEFAULT 1,          -- 1 active, 0 inactive
      blown INTEGER NOT NULL DEFAULT 0,           -- 1 means dead forever (Pro blow)

      resets_used INTEGER NOT NULL DEFAULT 0,
      purchase_paid_usd REAL NOT NULL DEFAULT 0,
      resets_paid_usd REAL NOT NULL DEFAULT 0,

      balance_usd REAL NOT NULL DEFAULT 0,        -- virtual
      withdrawable_usd REAL NOT NULL DEFAULT 0,   -- virtual, Pro only

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

      FOREIGN KEY(participant_id) REFERENCES participants(id),
      FOREIGN KEY(account_id) REFERENCES accounts(id)
    );
    """)

    # Tournament engine: manual match creation + resolution
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

    conn.commit()
    conn.close()

def init_db_with_migrations():
    """
    Create base tables if missing, and migrate columns if you ever update the app again.
    Since you asked to reset DB now, you likely won't need this,
    but leaving it here makes the app robust.
    """
    conn = db()
    cur = conn.cursor()

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

    # migrate rules
    safe_add_column(conn, "rules", "pro_threshold_usd REAL NOT NULL DEFAULT 9000")
    safe_add_column(conn, "rules", "cushion_usd REAL NOT NULL DEFAULT 4500")
    safe_add_column(conn, "rules", "max_pro_accounts INTEGER NOT NULL DEFAULT 5")

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
      active INTEGER NOT NULL DEFAULT 1,
      resets_used INTEGER NOT NULL DEFAULT 0,
      purchase_paid_usd REAL NOT NULL DEFAULT 0,
      resets_paid_usd REAL NOT NULL DEFAULT 0,
      balance_usd REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # migrate accounts
    safe_add_column(conn, "accounts", "phase TEXT NOT NULL DEFAULT 'Eval'")
    safe_add_column(conn, "accounts", "blown INTEGER NOT NULL DEFAULT 0")
    safe_add_column(conn, "accounts", "withdrawable_usd REAL NOT NULL DEFAULT 0")

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

    conn.commit()
    conn.close()

# =========================================================
# RULES
# =========================================================
def get_rules():
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

def set_rules(r):
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

# =========================================================
# ACCOUNTS
# =========================================================
def next_account_code(pid: int, pname: str) -> str:
    prefix = "".join([c for c in (pname or "").upper() if c.isalnum()])[:3] or f"P{pid}"
    conn = db()
    n = conn.execute("SELECT COUNT(*) FROM accounts WHERE participant_id=?", (pid,)).fetchone()[0]
    conn.close()
    return f"{prefix}-{n+1:04d}"

def buy_account(pid: int, pname: str, rules: dict):
    code = next_account_code(pid, pname)
    conn = db()
    conn.execute("""
      INSERT INTO accounts(participant_id, code, phase, active, blown, resets_used,
                           purchase_paid_usd, resets_paid_usd, balance_usd, withdrawable_usd, created_at)
      VALUES(?, ?, 'Eval', 1, 0, 0, ?, 0, 0, 0, ?)
    """, (pid, code, float(rules["account_cost_usd"]), now_iso()))
    conn.commit()
    conn.close()

def list_accounts_full():
    conn = db()
    df = pd.read_sql_query("""
      SELECT a.id,
             p.name AS participant,
             p.id   AS participant_id,
             a.code,
             a.phase,
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

def count_pro_active(rules: dict) -> int:
    conn = db()
    n = conn.execute("""
      SELECT COUNT(*) FROM accounts
      WHERE phase='Pro' AND blown=0
    """).fetchone()[0]
    conn.close()
    return int(n)

def recompute_withdrawable(account_id: int, rules: dict):
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT phase, balance_usd, blown FROM accounts WHERE id=?", (account_id,)).fetchone()
    if not row:
        conn.close()
        return
    phase, bal, blown = row[0], float(row[1]), int(row[2])

    if blown == 1 or phase != "Pro":
        wd = 0.0
    else:
        wd = max(0.0, bal - float(rules["cushion_usd"]))

    cur.execute("UPDATE accounts SET withdrawable_usd=? WHERE id=?", (wd, account_id))
    conn.commit()
    conn.close()

def can_promote_to_pro(account_id: int, rules: dict) -> (bool, str):
    """Manual promotion only, cap max_pro_accounts."""
    if count_pro_active(rules) >= rules["max_pro_accounts"]:
        return False, f"Pro cap reached ({rules['max_pro_accounts']})."
    conn = db()
    row = conn.execute("SELECT phase, balance_usd, blown FROM accounts WHERE id=?", (account_id,)).fetchone()
    conn.close()
    if not row:
        return False, "Account not found."
    phase, bal, blown = row[0], float(row[1]), int(row[2])
    if blown == 1:
        return False, "Account is blown."
    if phase != "Eval":
        return False, "Account is already Pro."
    if bal < rules["pro_threshold_usd"]:
        return False, f"Balance must be >= {rules['pro_threshold_usd']:,.0f} to promote."
    return True, "OK"

def promote_to_pro(account_id: int, rules: dict) -> str:
    ok, msg = can_promote_to_pro(account_id, rules)
    if not ok:
        return msg
    conn = db()
    conn.execute("UPDATE accounts SET phase='Pro' WHERE id=?", (account_id,))
    conn.commit()
    conn.close()
    recompute_withdrawable(account_id, rules)
    return "Promoted to Pro."

def manual_reset_eval(account_id: int, rules: dict) -> str:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("""
      SELECT phase, blown, active, resets_used
      FROM accounts WHERE id=?
    """, (account_id,)).fetchone()
    if not row:
        conn.close()
        return "Account not found."

    phase, blown, active, resets_used = row[0], int(row[1]), int(row[2]), int(row[3])

    if blown == 1:
        conn.close()
        return "Account is blown and cannot be reset."
    if phase != "Eval":
        conn.close()
        return "Pro accounts cannot be reset."
    if active == 1:
        conn.close()
        return "Account is already active."
    if resets_used >= rules["max_resets"]:
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

    recompute_withdrawable(account_id, rules)
    return f"Reset done. Used {resets_used}/{rules['max_resets']}."

# =========================================================
# WITHDRAWALS (REAL MONEY EVENTS YOU RECORD)
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
             w.note
      FROM withdrawals w
      JOIN participants p ON p.id=w.participant_id
      ORDER BY w.id DESC
    """, conn)
    conn.close()
    return df

def record_withdrawal(participant_id: int, account_id: int, percent: float, base_amount: float, note: str, rules: dict) -> str:
    amount = float(percent) * float(base_amount)
    conn = db()
    cur = conn.cursor()

    row = cur.execute("SELECT phase, blown, withdrawable_usd FROM accounts WHERE id=?", (account_id,)).fetchone()
    if not row:
        conn.close()
        return "Account not found."
    phase, blown, wd = row[0], int(row[1]), float(row[2])

    if blown == 1:
        conn.close()
        return "Account is blown."
    if phase != "Pro":
        conn.close()
        return "Only Pro accounts can withdraw."
    if amount > wd:
        conn.close()
        return f"Not enough withdrawable. Need {amount:,.2f}, available {wd:,.2f}"

    cur.execute("""
      INSERT INTO withdrawals(participant_id, account_id, percent, base_amount_usd, amount_usd, created_at, note)
      VALUES(?, ?, ?, ?, ?, ?, ?)
    """, (participant_id, account_id, float(percent), float(base_amount), float(amount), now_iso(), (note or "").strip()))

    # Reduce virtual balance to represent payout taken
    cur.execute("UPDATE accounts SET balance_usd = balance_usd - ? WHERE id=?", (float(amount), account_id))

    conn.commit()
    conn.close()

    recompute_withdrawable(account_id, rules)
    return f"Withdrawal recorded: {amount:,.2f} USD"

# =========================================================
# TOURNAMENT ENGINE (MANUAL MATCH SELECTION)
# =========================================================
def create_match(a_id: int, b_id: int, note: str = "") -> str:
    if a_id == b_id:
        return "Pick two different accounts."
    conn = db()
    # Ensure both exist and are eligible
    rowA = conn.execute("SELECT active, blown FROM accounts WHERE id=?", (a_id,)).fetchone()
    rowB = conn.execute("SELECT active, blown FROM accounts WHERE id=?", (b_id,)).fetchone()
    if not rowA or not rowB:
        conn.close()
        return "Account not found."
    if int(rowA[0]) != 1 or int(rowB[0]) != 1 or int(rowA[1]) == 1 or int(rowB[1]) == 1:
        conn.close()
        return "Both accounts must be Active and not Blown."
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

def resolve_match(match_id: int, winner_id: int, loser_id: int, rules: dict) -> str:
    win_amt = float(rules["win_amount_usd"])
    conn = db()
    cur = conn.cursor()

    # Winner +4500 virtual
    cur.execute("""
      UPDATE accounts
      SET balance_usd = balance_usd + ?
      WHERE id=?
    """, (win_amt, winner_id))

    # Loser -4500 virtual
    loser = cur.execute("SELECT phase, balance_usd FROM accounts WHERE id=?", (loser_id,)).fetchone()
    if not loser:
        conn.close()
        return "Loser account not found."
    loser_phase = loser[0]
    loser_balance_after = float(loser[1]) - win_amt

    if loser_phase == "Eval":
        # Eval loses => inactive until manual reset
        cur.execute("""
          UPDATE accounts
          SET balance_usd = balance_usd - ?,
              active = 0
          WHERE id=?
        """, (win_amt, loser_id))
    else:
        # Pro loses: blow only if below 0 (you can change to below cushion if you want)
        if loser_balance_after < 0:
            cur.execute("""
              UPDATE accounts
              SET balance_usd = balance_usd - ?,
                  active = 0,
                  blown = 1
              WHERE id=?
            """, (win_amt, loser_id))
        else:
            cur.execute("""
              UPDATE accounts
              SET balance_usd = balance_usd - ?
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

    # Recompute withdrawable for both
    recompute_withdrawable(winner_id, rules)
    recompute_withdrawable(loser_id, rules)

    return "Match resolved."

def matches_df(limit=50):
    conn = db()
    df = pd.read_sql_query(f"""
      SELECT *
      FROM matches
      ORDER BY id DESC
      LIMIT {int(limit)}
    """, conn)
    conn.close()
    return df

# =========================================================
# SUMMARY HELPERS
# =========================================================
def global_summary(acc: pd.DataFrame, wd: pd.DataFrame):
    if acc.empty:
        return {
            "accounts_total": 0,
            "spend": 0.0,
            "withdrawn": 0.0,
            "net": 0.0,
            "active": 0,
            "inactive": 0,
            "blown": 0,
            "eval_active": 0,
            "eval_inactive": 0,
            "pro_active": 0,
            "pro_blown": 0,
            "resets_used_total": 0,
        }

    spend = float((acc["purchase_paid_usd"].astype(float) + acc["resets_paid_usd"].astype(float)).sum())
    withdrawn = float(wd["amount_usd"].sum()) if not wd.empty else 0.0
    net = withdrawn - spend

    active = int((acc["active"].astype(int) == 1).sum())
    inactive = int((acc["active"].astype(int) == 0).sum())
    blown = int((acc["blown"].astype(int) == 1).sum())

    eval_active = int(((acc["phase"] == "Eval") & (acc["active"].astype(int) == 1) & (acc["blown"].astype(int) == 0)).sum())
    eval_inactive = int(((acc["phase"] == "Eval") & (acc["active"].astype(int) == 0) & (acc["blown"].astype(int) == 0)).sum())
    pro_active = int(((acc["phase"] == "Pro") & (acc["blown"].astype(int) == 0)).sum())
    pro_blown = int(((acc["phase"] == "Pro") & (acc["blown"].astype(int) == 1)).sum())
    resets_used_total = int(acc["resets_used"].astype(int).sum())

    return {
        "accounts_total": len(acc),
        "spend": spend,
        "withdrawn": withdrawn,
        "net": net,
        "active": active,
        "inactive": inactive,
        "blown": blown,
        "eval_active": eval_active,
        "eval_inactive": eval_inactive,
        "pro_active": pro_active,
        "pro_blown": pro_blown,
        "resets_used_total": resets_used_total,
    }

# =========================================================
# APP STARTUP
# =========================================================
# Use migrations init (works even if DB already exists). You asked to reset DB, but keep safe.
init_db_with_migrations()

st.title("Account Stats Game — Ledger + Manual Tournament Engine")

# =========================================================
# DB RESET PANEL (TOP)
# =========================================================
with st.expander("⚠️ Database reset", expanded=False):
    st.write("This deletes the local SQLite file and recreates an empty database.")
    confirm = st.checkbox("I understand this will delete all data in this app.", value=False)
    if st.button("RESET DATABASE NOW", type="primary", disabled=not confirm):
        try:
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            init_db_fresh()
            st.success("Database reset complete.")
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

# Load rules
rules = get_rules()

# =========================================================
# RULES UI
# =========================================================
with st.expander("Global Rules", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7 = st.columns(3)

    r = dict(rules)
    r["account_cost_usd"] = c1.number_input("Account cost (USD)", min_value=0.0, value=r["account_cost_usd"], step=1.0)
    r["reset_cost_usd"] = c2.number_input("Reset cost (USD)", min_value=0.0, value=r["reset_cost_usd"], step=1.0)
    r["max_resets"] = c3.number_input("Max resets (Eval)", min_value=0, value=r["max_resets"], step=1)
    r["max_pro_accounts"] = c4.number_input("Max Pro accounts (global)", min_value=0, value=r["max_pro_accounts"], step=1)

    r["win_amount_usd"] = c5.number_input("Win/Loss amount (virtual)", min_value=0.0, value=r["win_amount_usd"], step=100.0)
    r["pro_threshold_usd"] = c6.number_input("Pro threshold (virtual)", min_value=0.0, value=r["pro_threshold_usd"], step=500.0)
    r["cushion_usd"] = c7.number_input("Cushion (virtual, Pro)", min_value=0.0, value=r["cushion_usd"], step=100.0)

    if st.button("Save rules"):
        set_rules(r)
        st.success("Rules saved.")
        st.rerun()

# =========================================================
# DATA LOAD
# =========================================================
participants = list_participants()
accounts = list_accounts_full()
withdrawals = withdrawals_df()

# =========================================================
# GLOBAL SUMMARY
# =========================================================
s = global_summary(accounts, withdrawals)

st.subheader("Global Summary (clear + real money)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accounts", s["accounts_total"])
m2.metric("Spend (real)", f"{s['spend']:,.2f}")
m3.metric("Withdrawn (real)", f"{s['withdrawn']:,.2f}")
m4.metric("Net (real)", f"{s['net']:,.2f}")
m5.metric("Pro active", f"{s['pro_active']}/{rules['max_pro_accounts']}")

m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("Eval active", s["eval_active"])
m7.metric("Eval inactive", s["eval_inactive"])
m8.metric("Inactive total", s["inactive"])
m9.metric("Blown", s["blown"])
m10.metric("Resets used (total)", s["resets_used_total"])

st.divider()

# =========================================================
# PARTICIPANTS + LEDGER SECTION
# =========================================================
st.subheader("Ledger — Participants & Accounts")

left, right = st.columns([1.15, 2.85], gap="large")
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
        # Per participant summaries
        accounts["spend_usd"] = accounts["purchase_paid_usd"].astype(float) + accounts["resets_paid_usd"].astype(float)
        wd_by = withdrawals.groupby("participant_id", as_index=False)["amount_usd"].sum().rename(columns={"amount_usd": "withdrawn_usd"}) if not withdrawals.empty else pd.DataFrame(columns=["participant_id", "withdrawn_usd"])

        byp = accounts.groupby(["participant_id", "participant"], as_index=False).agg(
            accounts=("id", "count"),
            active=("active", "sum"),
            blown=("blown", "sum"),
            spend_usd=("spend_usd", "sum"),
            resets_used=("resets_used", "sum"),
        )
        byp = byp.merge(wd_by, on="participant_id", how="left").fillna({"withdrawn_usd": 0.0})
        byp["net_usd"] = byp["withdrawn_usd"] - byp["spend_usd"]
        byp = byp.sort_values("net_usd", ascending=False)

        st.markdown("### Per-participant summary (real money)")
        st.dataframe(byp[["participant", "accounts", "active", "blown", "resets_used", "spend_usd", "withdrawn_usd", "net_usd"]],
                     use_container_width=True, hide_index=True)

# Per participant detailed panels
if not participants.empty:
    for _, p in participants.iterrows():
        pid = int(p["id"])
        pname = str(p["name"])

        p_acc = accounts[accounts["participant_id"] == pid].copy()
        p_wd = withdrawals[withdrawals["participant_id"] == pid].copy() if not withdrawals.empty else pd.DataFrame()

        p_spend = float((p_acc["purchase_paid_usd"].astype(float) + p_acc["resets_paid_usd"].astype(float)).sum()) if not p_acc.empty else 0.0
        p_withdrawn = float(p_wd["amount_usd"].sum()) if not p_wd.empty else 0.0
        p_net = p_withdrawn - p_spend

        with st.expander(f"{pname} — Spend {p_spend:,.2f} | Withdrawn {p_withdrawn:,.2f} | Net {p_net:,.2f}", expanded=False):
            cA, cB, cC, cD = st.columns([1.1, 1.6, 1.6, 1.7], gap="large")

            # Buy
            with cA:
                st.markdown("#### Buy")
                if st.button(f"Buy account (${rules['account_cost_usd']:.0f})", key=f"buy_{pid}", use_container_width=True):
                    buy_account(pid, pname, rules)
                    st.rerun()

            # Reset (manual, Eval only)
            with cB:
                st.markdown("#### Manual reset (Eval only)")
                reset_candidates = p_acc[(p_acc["phase"] == "Eval") & (p_acc["active"] == 0) & (p_acc["blown"] == 0)].copy()
                if reset_candidates.empty:
                    st.caption("No inactive Eval accounts.")
                else:
                    reset_candidates["label"] = reset_candidates.apply(
                        lambda r: f"{r['code']} (#{int(r['id'])}) | resets {int(r['resets_used'])}/{rules['max_resets']} | bal {float(r['balance_usd']):,.0f}",
                        axis=1
                    )
                    sel = st.selectbox("Pick account", reset_candidates["label"].tolist(), key=f"reset_pick_{pid}")
                    acc_id = int(sel.split("#")[1].split(")")[0])
                    if st.button(f"Reset (+${rules['reset_cost_usd']:.0f})", key=f"reset_btn_{pid}", use_container_width=True):
                        msg = manual_reset_eval(acc_id, rules)
                        (st.success if msg.startswith("Reset") else st.warning)(msg)
                        st.rerun()

            # Promote to Pro (manual)
            with cC:
                st.markdown("#### Promote to Pro (manual)")
                eligible = p_acc[(p_acc["phase"] == "Eval") & (p_acc["blown"] == 0) & (p_acc["balance_usd"] >= rules["pro_threshold_usd"])].copy()
                if eligible.empty:
                    st.caption(f"No eligible accounts (need balance >= {rules['pro_threshold_usd']:,.0f}).")
                else:
                    eligible["label"] = eligible.apply(
                        lambda r: f"{r['code']} (#{int(r['id'])}) | bal {float(r['balance_usd']):,.0f}",
                        axis=1
                    )
                    selp = st.selectbox("Pick eligible", eligible["label"].tolist(), key=f"pro_pick_{pid}")
                    acc_id = int(selp.split("#")[1].split(")")[0])
                    if st.button("Promote to Pro", key=f"promote_{pid}", use_container_width=True):
                        msg = promote_to_pro(acc_id, rules)
                        (st.success if msg.startswith("Promoted") else st.warning)(msg)
                        st.rerun()

            # Record withdrawal (Pro only, manual)
            with cD:
                st.markdown("#### Record withdrawal (Pro only)")
                pro_accounts = p_acc[(p_acc["phase"] == "Pro") & (p_acc["blown"] == 0)].copy()
                if pro_accounts.empty:
                    st.caption("No Pro accounts.")
                else:
                    pro_accounts["label"] = pro_accounts.apply(
                        lambda r: f"{r['code']} (#{int(r['id'])}) | wd {float(r['withdrawable_usd']):,.0f} | bal {float(r['balance_usd']):,.0f}",
                        axis=1
                    )
                    selw = st.selectbox("Pick Pro", pro_accounts["label"].tolist(), key=f"wd_pick_{pid}")
                    acc_id = int(selw.split("#")[1].split(")")[0])

                    tier = st.selectbox("Tier", ["50%", "80%", "Custom"], key=f"wd_tier_{pid}")
                    if tier == "50%":
                        pct = 0.50
                    elif tier == "80%":
                        pct = 0.80
                    else:
                        pct = st.number_input("Percent", min_value=0.0, max_value=1.0, value=0.50, step=0.05, key=f"wd_pct_{pid}")

                    base = st.number_input("Base amount", min_value=0.0, value=float(rules["cushion_usd"]), step=100.0, key=f"wd_base_{pid}")
                    note = st.text_input("Note", key=f"wd_note_{pid}")

                    if st.button("Record withdrawal", key=f"wd_btn_{pid}", use_container_width=True):
                        msg = record_withdrawal(pid, acc_id, pct, base, note, rules)
                        (st.success if msg.startswith("Withdrawal") else st.warning)(msg)
                        st.rerun()

            st.markdown("#### Accounts")
            if p_acc.empty:
                st.write("No accounts yet.")
            else:
                view = p_acc.copy()
                view["spend_usd"] = view["purchase_paid_usd"].astype(float) + view["resets_paid_usd"].astype(float)
                view["status"] = view.apply(
                    lambda r: "BLOWN" if int(r["blown"]) == 1 else ("Active" if int(r["active"]) == 1 else "Inactive"),
                    axis=1
                )
                st.dataframe(
                    view[[
                        "id", "code", "phase", "status",
                        "resets_used",
                        "balance_usd", "withdrawable_usd",
                        "purchase_paid_usd", "resets_paid_usd", "spend_usd"
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

            if not p_wd.empty:
                st.markdown("#### Withdrawal history (real money records)")
                st.dataframe(
                    p_wd[["id", "account_id", "percent", "base_amount_usd", "amount_usd", "created_at", "note"]],
                    use_container_width=True,
                    hide_index=True
                )

st.divider()

# =========================================================
# TOURNAMENT ENGINE (SEPARATE SECTION)
# =========================================================
st.subheader("Tournament Engine — Manual Match Selection (you choose who fights)")

accounts = list_accounts_full()  # reload (in case something changed)
if accounts.empty:
    st.info("Create accounts first.")
else:
    eligible = accounts[(accounts["active"] == 1) & (accounts["blown"] == 0)].copy()
    eligible["label"] = eligible.apply(
        lambda r: f"{r['participant']} | {r['code']} (#{int(r['id'])}) | {r['phase']} | bal {float(r['balance_usd']):,.0f}",
        axis=1
    )

    col1, col2, col3 = st.columns([1.4, 1.4, 1.2], gap="large")

    with col1:
        a_pick = st.selectbox("Pick Account A", eligible["label"].tolist(), key="pick_A") if not eligible.empty else None
    with col2:
        b_pick = st.selectbox("Pick Account B", eligible["label"].tolist(), key="pick_B") if not eligible.empty else None
    with col3:
        note = st.text_input("Match note (optional)", key="match_note")

    def parse_id(label: str) -> int:
        return int(label.split("#")[1].split(")")[0])

    # Optional rule: prevent same-participant matches (toggle)
    prevent_same = st.checkbox("Prevent same participant fights", value=True)

    if st.button("Create match", type="primary"):
        if not a_pick or not b_pick:
            st.warning("Pick both accounts.")
        else:
            a_id = parse_id(a_pick)
            b_id = parse_id(b_pick)

            if prevent_same:
                pa = eligible[eligible["id"] == a_id]["participant_id"].iloc[0]
                pb = eligible[eligible["id"] == b_id]["participant_id"].iloc[0]
                if int(pa) == int(pb):
                    st.warning("Same participant. Pick a different opponent or disable the checkbox.")
                    st.stop()

            msg = create_match(a_id, b_id, note)
            (st.success if msg.startswith("Match created") else st.warning)(msg)
            st.rerun()

    open_m = get_open_match()
    if open_m is None:
        st.info("No open match. Create one above.")
    else:
        st.markdown("### Open match")
        a_id = int(open_m["a_account_id"])
        b_id = int(open_m["b_account_id"])

        idx = accounts.set_index("id")
        if a_id not in idx.index or b_id not in idx.index:
            st.warning("Match accounts missing. (Reset DB or delete match row.)")
        else:
            a = idx.loc[a_id]
            b = idx.loc[b_id]

            cA, cB = st.columns(2, gap="large")
            with cA:
                st.markdown(f"**A:** {a['participant']} — {a['code']} (#{a_id})")
                st.write(f"Phase: {a['phase']} | Balance: {float(a['balance_usd']):,.0f} | WD: {float(a['withdrawable_usd']):,.0f}")
                if st.button("✅ A WINS", type="primary", use_container_width=True):
                    msg = resolve_match(int(open_m["id"]), a_id, b_id, rules)
                    st.success(msg)
                    st.rerun()
            with cB:
                st.markdown(f"**B:** {b['participant']} — {b['code']} (#{b_id})")
                st.write(f"Phase: {b['phase']} | Balance: {float(b['balance_usd']):,.0f} | WD: {float(b['withdrawable_usd']):,.0f}")
                if st.button("✅ B WINS", type="primary", use_container_width=True):
                    msg = resolve_match(int(open_m["id"]), b_id, a_id, rules)
                    st.success(msg)
                    st.rerun()

st.divider()
st.subheader("Match history")
st.dataframe(matches_df(50), use_container_width=True, hide_index=True)

st.caption(
    "This is a tracking/stats game ledger: real money is only purchases/resets/recorded withdrawals; "
    "balances are simulated values used for your workflow tracking."
)
