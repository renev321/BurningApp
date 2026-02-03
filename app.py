import sqlite3
from datetime import datetime
import random
import string
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Account Fight Simulator", layout="wide")

DB_PATH = "account_game.db"

# -------------------- DB --------------------
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rules(
      id INTEGER PRIMARY KEY CHECK(id=1),
      account_cost_usd REAL NOT NULL DEFAULT 216,
      reset_cost_usd REAL NOT NULL DEFAULT 100,
      max_resets INTEGER NOT NULL DEFAULT 5,
      auto_delete_on_exhaust INTEGER NOT NULL DEFAULT 1
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
      code TEXT NOT NULL,                 -- auto name like A0001 / P1-0003
      active INTEGER NOT NULL DEFAULT 1,  -- 1 active, 0 inactive
      resets_used INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

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

    conn.commit()
    conn.close()

def get_rules():
    conn = db()
    row = conn.execute("""
      SELECT account_cost_usd, reset_cost_usd, max_resets, auto_delete_on_exhaust
      FROM rules WHERE id=1
    """).fetchone()
    conn.close()
    return {
        "account_cost_usd": float(row[0]),
        "reset_cost_usd": float(row[1]),
        "max_resets": int(row[2]),
        "auto_delete_on_exhaust": bool(row[3]),
    }

def set_rules(account_cost_usd, reset_cost_usd, max_resets, auto_delete_on_exhaust):
    conn = db()
    conn.execute("""
      UPDATE rules
      SET account_cost_usd=?, reset_cost_usd=?, max_resets=?, auto_delete_on_exhaust=?
      WHERE id=1
    """, (float(account_cost_usd), float(reset_cost_usd), int(max_resets), 1 if auto_delete_on_exhaust else 0))
    conn.commit()
    conn.close()

def list_participants():
    conn = db()
    df = pd.read_sql_query("SELECT id, name FROM participants ORDER BY name", conn)
    conn.close()
    return df

def add_participant(name: str):
    conn = db()
    conn.execute(
        "INSERT OR IGNORE INTO participants(name, created_at) VALUES(?, ?)",
        (name.strip(), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def delete_participant(pid: int):
    conn = db()
    conn.execute("DELETE FROM accounts WHERE participant_id=?", (int(pid),))
    conn.execute("DELETE FROM participants WHERE id=?", (int(pid),))
    conn.commit()
    conn.close()

def next_account_code(pid: int, pname: str) -> str:
    # e.g., REN-0001 or P2-0003
    prefix = "".join([c for c in pname.upper() if c.isalnum()])[:3]
    prefix = prefix if prefix else f"P{pid}"
    conn = db()
    n = conn.execute(
        "SELECT COUNT(*) FROM accounts WHERE participant_id=?",
        (int(pid),)
    ).fetchone()[0]
    conn.close()
    return f"{prefix}-{n+1:04d}"

def buy_account(pid: int, pname: str):
    code = next_account_code(pid, pname)
    conn = db()
    conn.execute("""
      INSERT INTO accounts(participant_id, code, active, resets_used, created_at)
      VALUES(?, ?, 1, 0, ?)
    """, (int(pid), code, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def list_accounts_full():
    conn = db()
    df = pd.read_sql_query("""
      SELECT a.id,
             p.name AS participant,
             a.code,
             a.active,
             a.resets_used,
             a.created_at
      FROM accounts a
      JOIN participants p ON p.id=a.participant_id
      ORDER BY p.name, a.created_at DESC
    """, conn)
    conn.close()
    return df

def toggle_account_active(account_id: int, active: bool):
    conn = db()
    conn.execute("UPDATE accounts SET active=? WHERE id=?", (1 if active else 0, int(account_id)))
    conn.commit()
    conn.close()

def delete_account(account_id: int):
    conn = db()
    conn.execute("DELETE FROM accounts WHERE id=?", (int(account_id),))
    conn.commit()
    conn.close()

def apply_loss_to_loser(account_id: int, rules):
    # increments resets_used; deactivates or deletes if exhausted
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT resets_used, active FROM accounts WHERE id=?", (int(account_id),)).fetchone()
    if not row:
        conn.close()
        return

    resets_used = int(row[0])
    if resets_used >= rules["max_resets"]:
        # already exhausted; enforce cleanup if desired
        if rules["auto_delete_on_exhaust"]:
            cur.execute("DELETE FROM accounts WHERE id=?", (int(account_id),))
        else:
            cur.execute("UPDATE accounts SET active=0 WHERE id=?", (int(account_id),))
        conn.commit()
        conn.close()
        return

    resets_used += 1
    if resets_used >= rules["max_resets"]:
        # exhausted now
        if rules["auto_delete_on_exhaust"]:
            cur.execute("DELETE FROM accounts WHERE id=?", (int(account_id),))
        else:
            cur.execute("UPDATE accounts SET resets_used=?, active=0 WHERE id=?", (resets_used, int(account_id)))
    else:
        cur.execute("UPDATE accounts SET resets_used=? WHERE id=?", (resets_used, int(account_id)))

    conn.commit()
    conn.close()

# -------------------- Matchmaking --------------------
def gen_session_name():
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rnd = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    return f"SESSION-{stamp}-{rnd}"

def pick_next_match(rules) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    # Choose 2 active accounts from different participants, with resets remaining
    df = list_accounts_full()
    if df.empty:
        return None, None

    df["resets_remaining"] = rules["max_resets"] - df["resets_used"]
    df = df[(df["active"] == 1) & (df["resets_remaining"] > 0)]
    if len(df) < 2:
        return None, None

    # ensure different participants
    participants = df["participant"].unique().tolist()
    if len(participants) < 2:
        return None, None

    # pick first participant randomly, then pick another different
    p1 = random.choice(participants)
    p2_choices = [p for p in participants if p != p1]
    p2 = random.choice(p2_choices)

    a = df[df["participant"] == p1].sample(1).iloc[0]
    b = df[df["participant"] == p2].sample(1).iloc[0]
    return a, b

def create_match(session_name: str, a_id: int, b_id: int):
    conn = db()
    conn.execute("""
      INSERT INTO matches(session_name, a_account_id, b_account_id, created_at)
      VALUES(?, ?, ?, ?)
    """, (session_name, int(a_id), int(b_id), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def resolve_match(match_id: int, winner_id: int, loser_id: int):
    conn = db()
    conn.execute("""
      UPDATE matches
      SET winner_account_id=?, loser_account_id=?, resolved_at=?
      WHERE id=?
    """, (int(winner_id), int(loser_id), datetime.utcnow().isoformat(), int(match_id)))
    conn.commit()
    conn.close()

def get_latest_open_match() -> pd.Series | None:
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

def accounts_by_id(ids: list[int]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    conn = db()
    q = f"""
      SELECT a.id, p.name AS participant, a.code, a.active, a.resets_used
      FROM accounts a
      JOIN participants p ON p.id=a.participant_id
      WHERE a.id IN ({",".join(["?"]*len(ids))})
    """
    df = pd.read_sql_query(q, conn, params=[int(x) for x in ids])
    conn.close()
    return df

# -------------------- UI --------------------
init_db()
rules = get_rules()

st.title("Account Fight Simulator (Stats Game)")

# RULES
with st.expander("Global Rules", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    account_cost = c1.number_input("Account cost (USD)", min_value=0.0, value=rules["account_cost_usd"], step=1.0)
    reset_cost = c2.number_input("Reset cost (USD)", min_value=0.0, value=rules["reset_cost_usd"], step=1.0)
    max_resets = c3.number_input("Max resets", min_value=0, value=rules["max_resets"], step=1)
    auto_del = c4.checkbox("Auto-delete when exhausted", value=rules["auto_delete_on_exhaust"])

    if st.button("Save rules"):
        set_rules(account_cost, reset_cost, int(max_resets), auto_del)
        st.success("Saved.")
        st.rerun()

rules = get_rules()

# SUMMARY
acc_df = list_accounts_full()
if not acc_df.empty:
    acc_df["resets_remaining"] = rules["max_resets"] - acc_df["resets_used"]
    total_accounts = len(acc_df)
    total_fees = total_accounts * rules["account_cost_usd"]
    total_resets_used = int(acc_df["resets_used"].sum())
    total_reset_cost = total_resets_used * rules["reset_cost_usd"]
    total_loss = total_fees + total_reset_cost
    active_count = int((acc_df["active"] == 1).sum())
else:
    total_accounts = active_count = 0
    total_fees = total_reset_cost = total_loss = 0.0
    total_resets_used = 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Participants", len(list_participants()))
m2.metric("Accounts", total_accounts)
m3.metric("Active accounts", active_count)
m4.metric("Total game cost (USD)", f"{total_loss:,.2f}")

st.divider()

# MAIN LAYOUT
left, mid, right = st.columns([1.2, 1.6, 1.8], gap="large")

# LEFT: Participants + Shop
with left:
    st.subheader("Participants")
    p_name = st.text_input("New participant", placeholder="Rene / Friend1 / Friend2")
    if st.button("Add participant", use_container_width=True):
        if p_name.strip():
            add_participant(p_name.strip())
            st.rerun()

    p_df = list_participants()
    if p_df.empty:
        st.info("Add participants to start.")
    else:
        st.subheader("Shop")
        pick = st.selectbox("Select participant", p_df["name"].tolist())
        pid = int(p_df[p_df["name"] == pick].iloc[0]["id"])

        if st.button(f"Buy account for {pick}", type="primary", use_container_width=True):
            buy_account(pid, pick)
            st.rerun()

        if st.button(f"Buy 2 accounts for {pick}", use_container_width=True):
            buy_account(pid, pick)
            buy_account(pid, pick)
            st.rerun()

        with st.expander("Danger zone"):
            if st.button(f"Delete participant {pick} (and all accounts)", use_container_width=True):
                delete_participant(pid)
                st.rerun()

# MID: Accounts table + actions
with mid:
    st.subheader("Accounts & Status")
    df = list_accounts_full()
    if df.empty:
        st.info("No accounts yet.")
    else:
        df["status"] = df["active"].map(lambda x: "Active" if x == 1 else "Inactive")
        df["resets_remaining"] = rules["max_resets"] - df["resets_used"]
        df["delete_ok"] = df["resets_used"] >= rules["max_resets"]

        show = df[["id", "participant", "code", "status", "resets_used", "resets_remaining", "delete_ok"]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.caption("Quick actions")
        a1, a2, a3 = st.columns(3)
        with a1:
            acc_id = st.number_input("Account ID", min_value=0, value=0, step=1)
        with a2:
            if st.button("Toggle Active/Inactive", use_container_width=True, disabled=(acc_id == 0)):
                # flip
                row = df[df["id"] == int(acc_id)]
                if not row.empty:
                    current = int(row.iloc[0]["active"])
                    toggle_account_active(int(acc_id), active=(current == 0))
                    st.rerun()
        with a3:
            if st.button("Delete account", use_container_width=True, disabled=(acc_id == 0)):
                delete_account(int(acc_id))
                st.rerun()

# RIGHT: Match Center (one-click flow)
with right:
    st.subheader("Match Center (Auto)")

    # Ensure we have an open match; if not, create one.
    open_match = get_latest_open_match()
    if open_match is None:
        a, b = pick_next_match(rules)
        if a is None or b is None:
            st.warning("Not enough eligible active accounts from different participants.")
        else:
            session = st.session_state.get("session_name")
            if not session:
                session = gen_session_name()
                st.session_state["session_name"] = session

            create_match(session, int(a["id"]), int(b["id"]))
            open_match = get_latest_open_match()

    if open_match is not None:
        st.write(f"Session: **{open_match['session_name']}**")
        st.write(f"Match created: {open_match['created_at']}")

        pair = accounts_by_id([int(open_match["a_account_id"]), int(open_match["b_account_id"])])
        if len(pair) < 2:
            st.warning("One of the accounts no longer exists. Creating a new match…")
            # mark this match as resolved with nulls (or just ignore) and rerun
            resolve_match(int(open_match["id"]), 0, 0)
            st.rerun()

        # Build UI cards
        pair = pair.set_index("id")
        a_id = int(open_match["a_account_id"])
        b_id = int(open_match["b_account_id"])

        a_row = pair.loc[a_id]
        b_row = pair.loc[b_id]

        a_resets_rem = rules["max_resets"] - int(a_row["resets_used"])
        b_resets_rem = rules["max_resets"] - int(b_row["resets_used"])

        cA, cB = st.columns(2, gap="large")

        with cA:
            st.markdown(f"### {a_row['participant']}")
            st.markdown(f"**{a_row['code']}**  (#{a_id})")
            st.write(f"Resets remaining: **{a_resets_rem}**")
            st.write(f"Status: **{'Active' if int(a_row['active'])==1 else 'Inactive'}**")

            if st.button("✅ A WINS", use_container_width=True, type="primary"):
                # loser is B
                resolve_match(int(open_match["id"]), a_id, b_id)
                apply_loss_to_loser(b_id, rules)
                # auto-create next match immediately
                st.rerun()

        with cB:
            st.markdown(f"### {b_row['participant']}")
            st.markdown(f"**{b_row['code']}**  (#{b_id})")
            st.write(f"Resets remaining: **{b_resets_rem}**")
            st.write(f"Status: **{'Active' if int(b_row['active'])==1 else 'Inactive'}**")

            if st.button("✅ B WINS", use_container_width=True, type="primary"):
                # loser is A
                resolve_match(int(open_match["id"]), b_id, a_id)
                apply_loss_to_loser(a_id, rules)
                st.rerun()

        st.divider()
        st.caption("Recent results")
        conn = db()
        hist = pd.read_sql_query("""
          SELECT id, session_name, a_account_id, b_account_id, winner_account_id, loser_account_id, resolved_at
          FROM matches
          WHERE resolved_at IS NOT NULL
          ORDER BY id DESC
          LIMIT 15
        """, conn)
        conn.close()
        st.dataframe(hist, use_container_width=True, hide_index=True)
