# app.py
import sqlite3
from datetime import datetime
import random
import string
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Account Simulator (Stats Game)", layout="wide")
DB_PATH = "account_game.db"

# =========================================================
# DB helpers
# =========================================================
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r[1] == col for r in rows)

def safe_add_column(conn: sqlite3.Connection, table: str, coldef: str):
    # coldef like: "phase TEXT NOT NULL DEFAULT 'Eval'"
    colname = coldef.split()[0]
    if not table_has_column(conn, table, colname):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef};")

def init_db():
    conn = db()
    cur = conn.cursor()

    # Rules (global knobs)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rules(
      id INTEGER PRIMARY KEY CHECK(id=1),
      account_cost_usd  REAL    NOT NULL DEFAULT 216,
      reset_cost_usd    REAL    NOT NULL DEFAULT 100,
      max_resets        INTEGER NOT NULL DEFAULT 5,
      win_amount_usd    REAL    NOT NULL DEFAULT 4500,
      pro_threshold_usd REAL    NOT NULL DEFAULT 9000,
      cushion_usd       REAL    NOT NULL DEFAULT 4500
    );
    """)
    cur.execute("INSERT OR IGNORE INTO rules(id) VALUES(1);")

    # Participants
    cur.execute("""
    CREATE TABLE IF NOT EXISTS participants(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );
    """)

    # Accounts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      code TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,          -- 1 active, 0 inactive
      resets_used INTEGER NOT NULL DEFAULT 0,
      purchase_paid_usd REAL NOT NULL DEFAULT 0, -- real spend
      resets_paid_usd REAL NOT NULL DEFAULT 0,   -- real spend
      balance_usd REAL NOT NULL DEFAULT 0,       -- virtual (game)
      created_at TEXT NOT NULL,
      FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # Migrations for accounts
    safe_add_column(conn, "accounts", "phase TEXT NOT NULL DEFAULT 'Eval'")            # Eval / Pro
    safe_add_column(conn, "accounts", "blown INTEGER NOT NULL DEFAULT 0")             # Pro dead forever
    safe_add_column(conn, "accounts", "withdrawable_usd REAL NOT NULL DEFAULT 0")     # virtual

    # Matches
    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_name TEXT NOT NULL,
      a_account_id INTEGER NOT NULL,
      b_account_id INTEGER NOT NULL,
      winner_account_id INTEGER NULL,
      loser_account_id INTEGER NULL,
      created_at TEXT NOT NULL,
      resolved_at TEXT NULL
    );
    """)

    # Withdrawals (real money events that you record manually)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS withdrawals(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      participant_id INTEGER NOT NULL,
      account_id INTEGER NOT NULL,
      percent REAL NOT NULL,          -- 0.50 / 0.80 etc
      base_amount_usd REAL NOT NULL,  -- usually 4500
      amount_usd REAL NOT NULL,       -- percent * base
      created_at TEXT NOT NULL,
      note TEXT NOT NULL DEFAULT '',
      FOREIGN KEY(participant_id) REFERENCES participants(id),
      FOREIGN KEY(account_id) REFERENCES accounts(id)
    );
    """)

    conn.commit()
    conn.close()

# =========================================================
# Rules
# =========================================================
def get_rules():
    conn = db()
    row = conn.execute("""
      SELECT account_cost_usd, reset_cost_usd, max_resets, win_amount_usd, pro_threshold_usd, cushion_usd
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
    }

def set_rules(account_cost_usd, reset_cost_usd, max_resets, win_amount_usd, pro_threshold_usd, cushion_usd):
    conn = db()
    conn.execute("""
      UPDATE rules
      SET account_cost_usd=?, reset_cost_usd=?, max_resets=?, win_amount_usd=?, pro_threshold_usd=?, cushion_usd=?
      WHERE id=1
    """, (
        float(account_cost_usd),
        float(reset_cost_usd),
        int(max_resets),
        float(win_amount_usd),
        float(pro_threshold_usd),
        float(cushion_usd),
    ))
    conn.commit()
    conn.close()

# =========================================================
# Participants
# =========================================================
def list_participants():
    conn = db()
    df = pd.read_sql_query("SELECT id, name FROM participants ORDER BY name", conn)
    conn.close()
    return df

def add_participant(name: str):
    name = name.strip()
    if not name:
        return
    conn = db()
    conn.execute(
        "INSERT OR IGNORE INTO participants(name, created_at) VALUES(?, ?)",
        (name, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def delete_participant(pid: int):
    conn = db()
    conn.execute("DELETE FROM withdrawals WHERE participant_id=?", (int(pid),))
    conn.execute("DELETE FROM accounts WHERE participant_id=?", (int(pid),))
    conn.execute("DELETE FROM participants WHERE id=?", (int(pid),))
    conn.commit()
    conn.close()

# =========================================================
# Accounts
# =========================================================
def next_account_code(pid: int, pname: str) -> str:
    prefix = "".join([c for c in pname.upper() if c.isalnum()])[:3]
    prefix = prefix if prefix else f"P{pid}"
    conn = db()
    n = conn.execute("SELECT COUNT(*) FROM accounts WHERE participant_id=?", (int(pid),)).fetchone()[0]
    conn.close()
    return f"{prefix}-{n+1:04d}"

def buy_account(pid: int, pname: str, rules: dict):
    code = next_account_code(pid, pname)
    conn = db()
    conn.execute("""
      INSERT INTO accounts(participant_id, code, active, resets_used, purchase_paid_usd, resets_paid_usd,
                           balance_usd, phase, blown, withdrawable_usd, created_at)
      VALUES(?, ?, 1, 0, ?, 0, 0, 'Eval', 0, 0, ?)
    """, (int(pid), code, float(rules["account_cost_usd"]), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def list_accounts_full():
    conn = db()
    df = pd.read_sql_query("""
      SELECT a.id,
             p.name AS participant,
             p.id   AS participant_id,
             a.code,
             a.active,
             a.phase,
             a.blown,
             a.resets_used,
             a.purchase_paid_usd,
             a.resets_paid_usd,
             a.balance_usd,
             a.withdrawable_usd,
             a.created_at
      FROM accounts a
      JOIN participants p ON p.id=a.participant_id
      ORDER BY p.name, a.created_at DESC
    """, conn)
    conn.close()
    return df

def delete_account(account_id: int):
    conn = db()
    conn.execute("DELETE FROM withdrawals WHERE account_id=?", (int(account_id),))
    conn.execute("DELETE FROM accounts WHERE id=?", (int(account_id),))
    conn.commit()
    conn.close()

def recompute_withdrawable(account_id: int, rules: dict):
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT phase, balance_usd, blown FROM accounts WHERE id=?", (int(account_id),)).fetchone()
    if not row:
        conn.close()
        return
    phase = row[0]
    bal = float(row[1])
    blown = int(row[2])

    if blown == 1:
        wd = 0.0
    elif phase == "Pro":
        wd = max(0.0, bal - float(rules["cushion_usd"]))
    else:
        wd = 0.0

    cur.execute("UPDATE accounts SET withdrawable_usd=? WHERE id=?", (wd, int(account_id)))
    conn.commit()
    conn.close()

def promote_if_needed(account_id: int, rules: dict):
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT phase, balance_usd, blown FROM accounts WHERE id=?", (int(account_id),)).fetchone()
    if not row:
        conn.close()
        return
    phase, bal, blown = row[0], float(row[1]), int(row[2])
    if blown == 1:
        conn.close()
        return
    if phase == "Eval" and bal >= float(rules["pro_threshold_usd"]):
        cur.execute("UPDATE accounts SET phase='Pro' WHERE id=?", (int(account_id),))
        conn.commit()
    conn.close()

def manual_reset_account(account_id: int, rules: dict) -> str:
    # Only Eval accounts can be reset, and only when inactive, and resets remaining
    conn = db()
    cur = conn.cursor()
    row = cur.execute("""
      SELECT phase, blown, active, resets_used
      FROM accounts
      WHERE id=?
    """, (int(account_id),)).fetchone()
    if not row:
        conn.close()
        return "Account not found."

    phase = row[0]
    blown = int(row[1])
    active = int(row[2])
    resets_used = int(row[3])

    if blown == 1:
        conn.close()
        return "This account is blown and cannot be reset."
    if phase != "Eval":
        conn.close()
        return "Pro accounts cannot be reset."
    if active == 1:
        conn.close()
        return "Account is already Active. Reset applies only to Inactive accounts."
    if resets_used >= int(rules["max_resets"]):
        conn.close()
        return f"No resets remaining (used {resets_used}/{rules['max_resets']})."

    resets_used += 1
    cur.execute("""
      UPDATE accounts
      SET resets_used=?,
          resets_paid_usd = resets_paid_usd + ?,
          active=1,
          balance_usd=0
      WHERE id=?
    """, (resets_used, float(rules["reset_cost_usd"]), int(account_id)))

    conn.commit()
    conn.close()

    recompute_withdrawable(account_id, rules)
    return f"Reset applied. Resets used: {resets_used}/{rules['max_resets']}"

# =========================================================
# Withdrawals (manual real-money records)
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
    row = cur.execute("""
      SELECT phase, blown, withdrawable_usd, balance_usd
      FROM accounts
      WHERE id=?
    """, (int(account_id),)).fetchone()
    if not row:
        conn.close()
        return "Account not found."
    phase = row[0]
    blown = int(row[1])
    withdrawable = float(row[2])
    balance = float(row[3])

    if blown == 1:
        conn.close()
        return "Account is blown."
    if phase != "Pro":
        conn.close()
        return "Only Pro accounts can record withdrawals."
    if amount > withdrawable:
        conn.close()
        return f"Not enough withdrawable. Need {amount:,.2f}, available {withdrawable:,.2f}"

    cur.execute("""
      INSERT INTO withdrawals(participant_id, account_id, percent, base_amount_usd, amount_usd, created_at, note)
      VALUES(?, ?, ?, ?, ?, ?, ?)
    """, (
        int(participant_id),
        int(account_id),
        float(percent),
        float(base_amount),
        float(amount),
        datetime.utcnow().isoformat(),
        (note or "").strip()
    ))

    # Reduce virtual balance to reflect payout (simulator)
    cur.execute("""
      UPDATE accounts
      SET balance_usd = balance_usd - ?
      WHERE id=?
    """, (float(amount), int(account_id)))

    conn.commit()
    conn.close()

    recompute_withdrawable(account_id, rules)
    return f"Recorded withdrawal: {amount:,.2f} USD"

# =========================================================
# Matchmaking / Matches
# =========================================================
def gen_session_name():
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rnd = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    return f"SESSION-{stamp}-{rnd}"

def get_latest_open_match():
    conn = db()
    df = pd.read_sql_query("""
      SELECT id, session_name, a_account_id, b_account_id, created_at
      FROM matches
      WHERE resolved_at IS NULL
      ORDER BY id DESC
      LIMIT 1
    """, conn)
    conn.close()
    if df.empty:
        return None
    return df.iloc[0]

def create_match(session_name: str, a_id: int, b_id: int):
    conn = db()
    conn.execute("""
      INSERT INTO matches(session_name, a_account_id, b_account_id, created_at)
      VALUES(?, ?, ?, ?)
    """, (session_name, int(a_id), int(b_id), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def pick_next_match(rules: dict) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    df = list_accounts_full()
    if df.empty:
        return None, None

    # Eligible: active, not blown
    df = df[(df["active"] == 1) & (df["blown"] == 0)]
    if len(df) < 2:
        return None, None

    participants = df["participant"].unique().tolist()
    if len(participants) < 2:
        return None, None

    p1 = random.choice(participants)
    p2 = random.choice([p for p in participants if p != p1])

    a = df[df["participant"] == p1].sample(1).iloc[0]
    b = df[df["participant"] == p2].sample(1).iloc[0]
    return a, b

def resolve_match_apply(match_id: int, winner_id: int, loser_id: int, rules: dict):
    win_amt = float(rules["win_amount_usd"])

    conn = db()
    cur = conn.cursor()

    # Winner gets +4500
    cur.execute("""
      UPDATE accounts
      SET balance_usd = balance_usd + ?,
          active = 1
      WHERE id=?
    """, (win_amt, int(winner_id)))

    # Loser gets -4500
    # - Eval: becomes inactive (manual reset needed)
    # - Pro: stays active unless balance would go below 0 -> blown
    loser_row = cur.execute("SELECT phase, balance_usd FROM accounts WHERE id=?", (int(loser_id),)).fetchone()
    if not loser_row:
        conn.close()
        return
    loser_phase = loser_row[0]
    loser_balance_before = float(loser_row[1])
    loser_balance_after = loser_balance_before - win_amt

    if loser_phase == "Eval":
        cur.execute("""
          UPDATE accounts
          SET balance_usd = balance_usd - ?,
              active = 0
          WHERE id=?
        """, (win_amt, int(loser_id)))
    else:
        # Pro
        if loser_balance_after < 0:
            # blown, no resets
            cur.execute("""
              UPDATE accounts
              SET balance_usd = balance_usd - ?,
                  active = 0,
                  blown = 1
              WHERE id=?
            """, (win_amt, int(loser_id)))
        else:
            # cushion absorbed it; remains active
            cur.execute("""
              UPDATE accounts
              SET balance_usd = balance_usd - ?,
                  active = 1
              WHERE id=?
            """, (win_amt, int(loser_id)))

    # Close match
    cur.execute("""
      UPDATE matches
      SET winner_account_id=?, loser_account_id=?, resolved_at=?
      WHERE id=?
    """, (int(winner_id), int(loser_id), datetime.utcnow().isoformat(), int(match_id)))

    conn.commit()
    conn.close()

    # Promotions + withdrawables
    promote_if_needed(winner_id, rules)
    promote_if_needed(loser_id, rules)
    recompute_withdrawable(winner_id, rules)
    recompute_withdrawable(loser_id, rules)

def recent_matches(limit=20):
    conn = db()
    df = pd.read_sql_query(f"""
      SELECT id, session_name, a_account_id, b_account_id, winner_account_id, loser_account_id, resolved_at
      FROM matches
      WHERE resolved_at IS NOT NULL
      ORDER BY id DESC
      LIMIT {int(limit)}
    """, conn)
    conn.close()
    return df

# =========================================================
# UI
# =========================================================
init_db()
rules = get_rules()

st.title("Account Simulator (Stats Game + Spend vs Withdrawals)")

# Global rules
with st.expander("Global Rules", expanded=True):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    account_cost = c1.number_input("Account cost", min_value=0.0, value=rules["account_cost_usd"], step=1.0)
    reset_cost = c2.number_input("Reset cost", min_value=0.0, value=rules["reset_cost_usd"], step=1.0)
    max_resets = c3.number_input("Max resets", min_value=0, value=rules["max_resets"], step=1)
    win_amt = c4.number_input("Win/Loss amount", min_value=0.0, value=rules["win_amount_usd"], step=100.0)
    pro_thr = c5.number_input("Pro threshold", min_value=0.0, value=rules["pro_threshold_usd"], step=500.0)
    cushion = c6.number_input("Cushion", min_value=0.0, value=rules["cushion_usd"], step=100.0)

    if st.button("Save rules"):
        set_rules(account_cost, reset_cost, int(max_resets), win_amt, pro_thr, cushion)
        st.success("Saved.")
        st.rerun()

rules = get_rules()
st.divider()

# Add participant
l0, r0 = st.columns([1.1, 2.9], gap="large")
with l0:
    st.subheader("Add participant")
    name = st.text_input("Name", placeholder="Rene / Friend1 / Friend2")
    if st.button("Add", use_container_width=True):
        add_participant(name)
        st.rerun()

with r0:
    st.subheader("Global summary")
    acc = list_accounts_full()
    wd = withdrawals_df()

    if acc.empty:
        st.info("No accounts yet.")
    else:
        acc["spend_usd"] = acc["purchase_paid_usd"] + acc["resets_paid_usd"]
        total_spend = float(acc["spend_usd"].sum())
        total_withdrawn = float(wd["amount_usd"].sum()) if not wd.empty else 0.0
        net = total_withdrawn - total_spend
        roi = (net / total_spend) if total_spend > 0 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accounts", len(acc))
        m2.metric("Total spend (real)", f"{total_spend:,.2f}")
        m3.metric("Total withdrawn (real)", f"{total_withdrawn:,.2f}")
        m4.metric("Net (real)", f"{net:,.2f}")
        m5.metric("ROI", f"{roi*100:,.1f}%")

        byp = acc.groupby("participant", as_index=False).agg(
            accounts=("id", "count"),
            active=("active", "sum"),
            blown=("blown", "sum"),
            spend_usd=("spend_usd", "sum"),
            balance_virtual=("balance_usd", "sum"),
            withdrawable_virtual=("withdrawable_usd", "sum"),
        )
        if not wd.empty:
            wd_by = wd.groupby("participant", as_index=False)["amount_usd"].sum().rename(columns={"amount_usd": "withdrawn_usd"})
            byp = byp.merge(wd_by, on="participant", how="left").fillna(0.0)
        else:
            byp["withdrawn_usd"] = 0.0
        byp["net_usd"] = byp["withdrawn_usd"] - byp["spend_usd"]
        byp = byp.sort_values("net_usd", ascending=False)

        st.dataframe(byp, use_container_width=True, hide_index=True)

st.divider()

# Match center
st.subheader("Match Center (auto-match, you click winner)")

session = st.session_state.get("session_name")
if not session:
    session = gen_session_name()
    st.session_state["session_name"] = session

open_match = get_latest_open_match()
if open_match is None:
    a, b = pick_next_match(rules)
    if a is None or b is None:
        st.warning("Need at least 2 ACTIVE accounts from different participants (not blown).")
    else:
        create_match(session, int(a["id"]), int(b["id"]))
        open_match = get_latest_open_match()

if open_match is not None:
    st.write(f"Session: **{open_match['session_name']}**")
    a_id = int(open_match["a_account_id"])
    b_id = int(open_match["b_account_id"])

    acc_now = list_accounts_full().set_index("id")
    if a_id not in acc_now.index or b_id not in acc_now.index:
        st.warning("Matched account missing. Try again by refreshing.")
    else:
        a_row = acc_now.loc[a_id]
        b_row = acc_now.loc[b_id]

        cA, cB = st.columns(2, gap="large")

        def card(col, row, acc_id: int, label: str):
            with col:
                st.markdown(f"### {row['participant']}")
                st.markdown(f"**{row['code']}** (#{acc_id})")
                st.write(f"Phase: **{row['phase']}**")
                st.write(f"Status: **{'Active' if int(row['active'])==1 else 'Inactive'}**   |   Blown: **{'Yes' if int(row['blown'])==1 else 'No'}**")
                st.write(f"Resets used: **{int(row['resets_used'])}/{rules['max_resets']}**")
                st.write(f"Balance (virtual): **{float(row['balance_usd']):,.0f}**")
                st.write(f"Withdrawable (virtual): **{float(row['withdrawable_usd']):,.0f}**")
                return st.button(f"✅ {label} WINS", type="primary", use_container_width=True)

        a_wins = card(cA, a_row, a_id, "A")
        b_wins = card(cB, b_row, b_id, "B")

        if a_wins:
            resolve_match_apply(int(open_match["id"]), a_id, b_id, rules)
            st.rerun()
        if b_wins:
            resolve_match_apply(int(open_match["id"]), b_id, a_id, rules)
            st.rerun()

st.divider()

# Per-user tables
st.subheader("Participants (each has their own table)")

p_df = list_participants()
acc_df = list_accounts_full()
wd_df = withdrawals_df()

if p_df.empty:
    st.info("Add participants above.")
else:
    for _, prow in p_df.iterrows():
        pid = int(prow["id"])
        pname = str(prow["name"])

        user_acc = acc_df[acc_df["participant_id"] == pid].copy()
        user_acc["status"] = user_acc["active"].map(lambda x: "Active" if int(x) == 1 else "Inactive")
        user_acc["resets_remaining"] = rules["max_resets"] - user_acc["resets_used"]
        user_acc["spend_usd"] = user_acc["purchase_paid_usd"] + user_acc["resets_paid_usd"]

        user_wd = wd_df[wd_df["participant_id"] == pid].copy() if not wd_df.empty else pd.DataFrame()
        total_spend_u = float(user_acc["spend_usd"].sum()) if not user_acc.empty else 0.0
        total_withdrawn_u = float(user_wd["amount_usd"].sum()) if not user_wd.empty else 0.0
        net_u = total_withdrawn_u - total_spend_u

        with st.expander(f"{pname}", expanded=True):
            u1, u2, u3, u4 = st.columns([1.2, 1.2, 2.2, 1.6], gap="large")

            # Shop
            with u1:
                st.markdown("#### Shop")
                if st.button(f"Buy account ({rules['account_cost_usd']:.0f} USD)", key=f"buy_{pid}", use_container_width=True):
                    buy_account(pid, pname, rules)
                    st.rerun()

            # Summary
            with u2:
                st.markdown("#### Summary")
                st.write(f"Accounts: **{len(user_acc)}**")
                st.write(f"Spend (real): **{total_spend_u:,.2f}**")
                st.write(f"Withdrawn (real): **{total_withdrawn_u:,.2f}**")
                st.write(f"Net (real): **{net_u:,.2f}**")

            # Manual reset (Eval only)
            with u3:
                st.markdown("#### Manual reset (Eval only)")
                inactive_eval = user_acc[(user_acc["active"] == 0) & (user_acc["phase"] == "Eval") & (user_acc["blown"] == 0)].copy()
                if inactive_eval.empty:
                    st.info("No inactive Eval accounts to reset.")
                else:
                    inactive_eval["label"] = inactive_eval.apply(
                        lambda r: f"{r['code']} (#{int(r['id'])}) | resets {int(r['resets_used'])}/{rules['max_resets']} | bal {float(r['balance_usd']):,.0f}",
                        axis=1
                    )
                    pick = st.selectbox("Select inactive Eval account", inactive_eval["label"].tolist(), key=f"pick_reset_{pid}")
                    acc_id = int(pick.split("#")[1].split(")")[0])
                    if st.button(f"Reset selected (+{rules['reset_cost_usd']:.0f} USD)", key=f"do_reset_{pid}", type="primary", use_container_width=True):
                        msg = manual_reset_account(acc_id, rules)
                        if msg.startswith("Reset applied"):
                            st.success(msg)
                        else:
                            st.warning(msg)
                        st.rerun()

            # Withdrawals + cleanup
            with u4:
                st.markdown("#### Payout record (Pro only)")
                pro = user_acc[(user_acc["phase"] == "Pro") & (user_acc["blown"] == 0)].copy()
                if pro.empty:
                    st.info("No Pro accounts.")
                else:
                    pro["label"] = pro.apply(
                        lambda r: f"{r['code']} (#{int(r['id'])}) | wd {float(r['withdrawable_usd']):,.0f} | bal {float(r['balance_usd']):,.0f}",
                        axis=1
                    )
                    pickp = st.selectbox("Select Pro account", pro["label"].tolist(), key=f"pick_pro_{pid}")
                    pro_id = int(pickp.split("#")[1].split(")")[0])
                    base = st.number_input("Base amount", min_value=0.0, value=float(rules["cushion_usd"]), step=100.0, key=f"base_{pid}")
                    tier = st.selectbox("Percent", ["50%", "80%"], key=f"tier_{pid}")
                    pct = 0.50 if tier == "50%" else 0.80
                    note = st.text_input("Note (optional)", key=f"note_{pid}")

                    if st.button(f"Record withdrawal ({int(pct*100)}% of {base:,.0f})", key=f"wd_{pid}", use_container_width=True):
                        msg = record_withdrawal(pid, pro_id, pct, base, note, rules)
                        if msg.startswith("Recorded"):
                            st.success(msg)
                        else:
                            st.warning(msg)
                        st.rerun()

                st.markdown("#### Cleanup")
                # Exhausted Eval accounts OR blown Pro accounts
                exhausted_eval = user_acc[(user_acc["phase"] == "Eval") & (user_acc["resets_used"] >= rules["max_resets"])].copy()
                blown_pro = user_acc[(user_acc["phase"] == "Pro") & (user_acc["blown"] == 1)].copy()
                cleanup = pd.concat([exhausted_eval, blown_pro], ignore_index=True) if (not exhausted_eval.empty or not blown_pro.empty) else pd.DataFrame()

                if cleanup.empty:
                    st.caption("Nothing to clean.")
                else:
                    cleanup["label"] = cleanup.apply(
                        lambda r: f"{r['code']} (#{int(r['id'])}) | {r['phase']} | {'BLOWN' if int(r['blown'])==1 else 'EXHAUSTED'}",
                        axis=1
                    )
                    pickc = st.selectbox("Delete account", cleanup["label"].tolist(), key=f"pick_clean_{pid}")
                    del_id = int(pickc.split("#")[1].split(")")[0])
                    if st.button("Delete selected", key=f"del_{pid}", use_container_width=True):
                        delete_account(del_id)
                        st.rerun()

            st.markdown("#### Accounts table")
            if user_acc.empty:
                st.write("No accounts yet.")
            else:
                view = user_acc[[
                    "id", "code", "phase", "status", "blown",
                    "resets_used", "resets_remaining",
                    "balance_usd", "withdrawable_usd",
                    "purchase_paid_usd", "resets_paid_usd", "spend_usd"
                ]].copy()
                st.dataframe(view, use_container_width=True, hide_index=True)

            if not user_wd.empty:
                st.markdown("#### Withdrawal history")
                st.dataframe(
                    user_wd[["id", "account_id", "percent", "base_amount_usd", "amount_usd", "created_at", "note"]],
                    use_container_width=True,
                    hide_index=True
                )

st.divider()
st.subheader("Recent matches")
st.dataframe(recent_matches(25), use_container_width=True, hide_index=True)

st.caption(
    "This app is a stats/game ledger: real spend (purchases/resets) and recorded withdrawals are tracked, "
    "while balances/thresholds/cushion are simulated values to support your tracking."
)
