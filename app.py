import json
import random
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Bracket Nodes + Cost Simulator", layout="wide")
DB_PATH = "bracket_nodes.db"

# =========================================================
# DB
# =========================================================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

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
        nickname TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (participant_id) REFERENCES participants(id)
    );
    """)

    # Global rules stored once
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rules(
        id INTEGER PRIMARY KEY CHECK (id = 1),
        account_fee_usd REAL NOT NULL DEFAULT 216,
        reset_cost_usd REAL NOT NULL DEFAULT 100,
        max_resets INTEGER NOT NULL DEFAULT 5
    );
    """)

    cur.execute("""
    INSERT OR IGNORE INTO rules(id, account_fee_usd, reset_cost_usd, max_resets)
    VALUES(1, 216, 100, 5);
    """)

    # Tournament + State
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tournaments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tournament_state(
        tournament_id INTEGER PRIMARY KEY,
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()

def get_rules() -> Dict[str, Any]:
    conn = db()
    row = conn.execute("SELECT account_fee_usd, reset_cost_usd, max_resets FROM rules WHERE id=1").fetchone()
    conn.close()
    return {"account_fee_usd": row[0], "reset_cost_usd": row[1], "max_resets": row[2]}

def set_rules(account_fee_usd: float, reset_cost_usd: float, max_resets: int):
    conn = db()
    conn.execute(
        "UPDATE rules SET account_fee_usd=?, reset_cost_usd=?, max_resets=? WHERE id=1",
        (float(account_fee_usd), float(reset_cost_usd), int(max_resets))
    )
    conn.commit()
    conn.close()

def upsert_participant(name: str):
    conn = db()
    conn.execute(
        "INSERT OR IGNORE INTO participants(name, created_at) VALUES(?, ?)",
        (name.strip(), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def list_participants() -> pd.DataFrame:
    conn = db()
    df = pd.read_sql_query("SELECT id, name FROM participants ORDER BY name", conn)
    conn.close()
    return df

def add_account(participant_name: str, nickname: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM participants WHERE name=?", (participant_name.strip(),))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError("Participant not found.")
    pid = row[0]
    conn.execute(
        "INSERT INTO accounts(participant_id, nickname, created_at) VALUES(?, ?, ?)",
        (pid, nickname.strip(), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def list_accounts() -> pd.DataFrame:
    conn = db()
    df = pd.read_sql_query("""
    SELECT a.id, p.name AS participant, a.nickname
    FROM accounts a
    JOIN participants p ON p.id = a.participant_id
    ORDER BY p.name, a.nickname
    """, conn)
    conn.close()
    return df

def create_tournament(name: str) -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tournaments(name, created_at) VALUES(?, ?)",
        (name.strip(), datetime.utcnow().isoformat())
    )
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return tid

def list_tournaments() -> pd.DataFrame:
    conn = db()
    df = pd.read_sql_query("SELECT id, name, created_at FROM tournaments ORDER BY id DESC", conn)
    conn.close()
    return df

def save_state(tournament_id: int, state: dict):
    conn = db()
    conn.execute("""
    INSERT INTO tournament_state(tournament_id, state_json, updated_at)
    VALUES(?, ?, ?)
    ON CONFLICT(tournament_id) DO UPDATE SET
        state_json=excluded.state_json,
        updated_at=excluded.updated_at
    """, (int(tournament_id), json.dumps(state), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def load_state(tournament_id: int) -> Optional[dict]:
    conn = db()
    row = conn.execute("SELECT state_json FROM tournament_state WHERE tournament_id=?", (int(tournament_id),)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

# =========================================================
# Bracket State (single elimination) + Cost tracking
# =========================================================
def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p

def seed_bracket(accounts: List[dict], shuffle=True) -> dict:
    players = accounts[:]
    if shuffle:
        random.shuffle(players)

    n = len(players)
    size = next_pow2(n)
    slots = players + [None] * (size - n)

    matches = []
    match_id = 1
    for i in range(0, size, 2):
        a = slots[i]
        b = slots[i+1]
        matches.append({
            "match_id": match_id,
            "a": a,
            "b": b,
            "winner_id": None,
            "loser_id": None,
            "note": "",
        })
        match_id += 1

    state = {
        "created_at": datetime.utcnow().isoformat(),
        "round": 1,
        "matches": matches,
        "history": [],  # completed rounds
        # per-account simulation ledger
        "ledger": {
            # account_id: { "resets_used": 0, "status": "Active", "loss_usd": fee+resets }
        },
        "completed": False
    }
    return state

def init_ledger_if_missing(state: dict, accounts: List[dict], rules: dict):
    ledger = state.get("ledger", {})
    for a in accounts:
        aid = str(a["id"])
        if aid not in ledger:
            ledger[aid] = {
                "resets_used": 0,
                "status": "Active",  # Active / Exhausted
                "loss_usd": float(rules["account_fee_usd"]),  # fee counts once
                "participant": a["participant"],
                "nickname": a["nickname"]
            }
    state["ledger"] = ledger

def account_alive(state: dict, account_id: int) -> bool:
    rec = state["ledger"].get(str(account_id))
    return rec is not None and rec["status"] == "Active"

def apply_loss_to_loser(state: dict, loser_id: int, rules: dict):
    rec = state["ledger"][str(loser_id)]
    if rec["status"] != "Active":
        return

    if rec["resets_used"] < int(rules["max_resets"]):
        rec["resets_used"] += 1
        rec["loss_usd"] += float(rules["reset_cost_usd"])
        # still active unless we just hit limit and you want to mark exhausted at max
        if rec["resets_used"] >= int(rules["max_resets"]):
            rec["status"] = "Exhausted"
    else:
        rec["status"] = "Exhausted"

def build_next_round(state: dict):
    # only if all matches have a winner (or BYE)
    for m in state["matches"]:
        a = m["a"]; b = m["b"]
        if a is None and b is None:
            m["winner_id"] = None
        elif a is not None and b is None:
            m["winner_id"] = a["id"]
        elif a is None and b is not None:
            m["winner_id"] = b["id"]
        else:
            if m["winner_id"] is None:
                return state  # not ready

    winners = []
    for m in state["matches"]:
        if m["winner_id"] is not None:
            # find winner obj
            if m["a"] is not None and m["a"]["id"] == m["winner_id"]:
                winners.append(m["a"])
            elif m["b"] is not None and m["b"]["id"] == m["winner_id"]:
                winners.append(m["b"])

    # archive current round
    state["history"].append({"round": state["round"], "matches": state["matches"]})

    if len(winners) <= 1:
        state["completed"] = True
        state["matches"] = []
        return state

    # build next round
    next_matches = []
    match_id = 1
    for i in range(0, len(winners), 2):
        a = winners[i]
        b = winners[i+1] if i+1 < len(winners) else None
        next_matches.append({
            "match_id": match_id,
            "a": a,
            "b": b,
            "winner_id": None if (a and b) else (a["id"] if a else None),
            "loser_id": None,
            "note": "",
        })
        match_id += 1

    state["round"] += 1
    state["matches"] = next_matches
    return state

# =========================================================
# Graph rendering (Nodes)
# =========================================================
def make_graph_for_round(state: dict) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []

    # round title node
    nodes.append(Node(id=f"R{state['round']}", label=f"Round {state['round']}", size=18))

    for m in state["matches"]:
        mid = m["match_id"]
        match_node_id = f"M{state['round']}_{mid}"
        nodes.append(Node(
            id=match_node_id,
            label=f"Match {mid}",
            size=22
        ))
        edges.append(Edge(source=f"R{state['round']}", target=match_node_id))

        def add_acc_node(acc, side_label: str):
            aid = acc["id"]
            led = state["ledger"].get(str(aid), {})
            status = led.get("status", "Active")
            resets = led.get("resets_used", 0)

            label = f"{acc['participant']}\n{acc['nickname']} (#{aid})\n{status} | resets {resets}"
            nid = f"A{aid}"
            # Node already exists? keep unique
            if not any(n.id == nid for n in nodes):
                nodes.append(Node(id=nid, label=label, size=18))
            edges.append(Edge(source=nid, target=match_node_id, label=side_label))

        if m["a"] is not None:
            add_acc_node(m["a"], "A")
        else:
            bye_id = f"BYE_A_{state['round']}_{mid}"
            nodes.append(Node(id=bye_id, label="BYE", size=14))
            edges.append(Edge(source=bye_id, target=match_node_id, label="A"))

        if m["b"] is not None:
            add_acc_node(m["b"], "B")
        else:
            bye_id = f"BYE_B_{state['round']}_{mid}"
            nodes.append(Node(id=bye_id, label="BYE", size=14))
            edges.append(Edge(source=bye_id, target=match_node_id, label="B"))

    return nodes, edges

# =========================================================
# App
# =========================================================
init_db()

st.title("Bracket Nodes + Cost Simulator (Stats Game)")

rules = get_rules()

# ---- Global rules panel
with st.expander("Global Rules (apply to everyone)", expanded=True):
    c1, c2, c3 = st.columns(3)
    fee = c1.number_input("Account fee (USD)", min_value=0.0, value=float(rules["account_fee_usd"]), step=1.0)
    reset_cost = c2.number_input("Reset cost (USD)", min_value=0.0, value=float(rules["reset_cost_usd"]), step=1.0)
    max_resets = c3.number_input("Max resets per account", min_value=0, value=int(rules["max_resets"]), step=1)

    if st.button("Save rules"):
        set_rules(fee, reset_cost, int(max_resets))
        st.success("Rules saved.")
        st.rerun()

st.divider()

# ---- Setup: participants + accounts
left, right = st.columns([2, 3], gap="large")

with left:
    st.subheader("Participants")
    name = st.text_input("Add participant", placeholder="Rene / Friend1 / Friend2")
    if st.button("Add participant", use_container_width=True):
        if name.strip():
            upsert_participant(name.strip())
            st.rerun()

    st.subheader("Accounts")
    p_df = list_participants()
    if len(p_df) == 0:
        st.info("Add at least one participant.")
    else:
        p = st.selectbox("Participant", p_df["name"].tolist())
        nick = st.text_input("Account nickname", placeholder="A1 / A2 / B1 ...")
        if st.button("Add account", use_container_width=True):
            if nick.strip():
                add_account(p, nick.strip())
                st.rerun()

with right:
    st.subheader("Current accounts")
    accounts_df = list_accounts()
    st.dataframe(accounts_df, use_container_width=True, hide_index=True)

st.divider()

# ---- Tournament controls
tcol1, tcol2 = st.columns([2, 3], gap="large")

with tcol1:
    st.subheader("Tournament")
    tname = st.text_input("New tournament name", placeholder="Feb-Week1")
    shuffle = st.checkbox("Shuffle seeds", value=True)
    if st.button("Create tournament from ALL accounts", use_container_width=True, disabled=(len(accounts_df) < 2 or len(tname.strip()) == 0)):
        tid = create_tournament(tname.strip())
        state = seed_bracket(accounts_df.to_dict(orient="records"), shuffle=shuffle)
        init_ledger_if_missing(state, accounts_df.to_dict(orient="records"), get_rules())
        save_state(tid, state)
        st.session_state["tid"] = tid
        st.rerun()

    tdf = list_tournaments()
    if len(tdf):
        label_map = {f"#{int(r.id)} • {r.name} ({r.created_at[:10]})": int(r.id) for r in tdf.itertuples()}
        pick = st.selectbox("Open existing", list(label_map.keys()))
        if st.button("Open selected", type="secondary", use_container_width=True):
            st.session_state["tid"] = label_map[pick]
            st.rerun()
    else:
        st.info("No tournaments yet.")

with tcol2:
    tid = st.session_state.get("tid")
    if not tid:
        st.info("Create or open a tournament.")
    else:
        state = load_state(tid)
        if not state:
            st.warning("No saved state.")
        else:
            # ensure ledger exists for all accounts ever used
            all_accounts = accounts_df.to_dict(orient="records")
            init_ledger_if_missing(state, all_accounts, get_rules())

            # ---- Summary (loss/cost)
            ledger = state["ledger"]
            df_led = pd.DataFrame([{
                "account_id": int(aid),
                "participant": rec["participant"],
                "nickname": rec["nickname"],
                "status": rec["status"],
                "resets_used": rec["resets_used"],
                "loss_usd": rec["loss_usd"],
            } for aid, rec in ledger.items()])

            # Totals
            total_loss = float(df_led["loss_usd"].sum()) if len(df_led) else 0.0
            total_resets_cost = float(df_led["resets_used"].sum()) * float(get_rules()["reset_cost_usd"])
            total_fees = len(df_led) * float(get_rules()["account_fee_usd"])

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Accounts in ledger", len(df_led))
            s2.metric("Total Fees (USD)", f"{total_fees:,.2f}")
            s3.metric("Total Reset Cost (USD)", f"{total_resets_cost:,.2f}")
            s4.metric("Total Loss (USD)", f"{total_loss:,.2f}")

            st.caption("Per participant")
            byp = df_led.groupby("participant", as_index=False).agg(
                accounts=("account_id", "count"),
                loss_usd=("loss_usd", "sum"),
                resets_used=("resets_used", "sum"),
            ).sort_values("loss_usd", ascending=False)
            st.dataframe(byp, use_container_width=True, hide_index=True)

            st.divider()

            # ---- Node graph for current round
            if state.get("completed"):
                st.success("Tournament completed.")
                st.dataframe(df_led.sort_values(["participant","nickname"]), use_container_width=True, hide_index=True)
            else:
                st.subheader(f"Round {state['round']} – Node Bracket")
                nodes, edges = make_graph_for_round(state)

                config = Config(
                    width="100%",
                    height=520,
                    directed=True,
                    physics=False,
                    hierarchical=True
                )

                selected = agraph(nodes=nodes, edges=edges, config=config)
                st.caption("Click an account node, then pick the match and set winner.")

                # Controls to set winner
                matches = state["matches"]
                match_options = []
                for m in matches:
                    a = m["a"]; b = m["b"]
                    if a is None or b is None:
                        continue
                    match_options.append(f"Match {m['match_id']}: A#{a['id']} vs B#{b['id']}")
                if len(match_options) == 0:
                    st.info("No selectable matches (BYEs only). Click **Build next round**.")
                else:
                    sel_match_label = st.selectbox("Select match to resolve", match_options)
                    match_id = int(sel_match_label.split(":")[0].split()[1])

                    # Parse selected node id -> account id
                    selected_account_id = None
                    if isinstance(selected, dict) and "id" in selected and str(selected["id"]).startswith("A"):
                        try:
                            selected_account_id = int(str(selected["id"])[1:])
                        except:
                            selected_account_id = None

                    st.write(f"Selected node account id: **{selected_account_id if selected_account_id else '—'}**")

                    if st.button("Set selected node as winner", type="primary", use_container_width=True, disabled=(selected_account_id is None)):
                        rules_now = get_rules()
                        # find match
                        for m in state["matches"]:
                            if m["match_id"] != match_id:
                                continue
                            a = m["a"]; b = m["b"]
                            if not a or not b:
                                continue
                            if selected_account_id not in (a["id"], b["id"]):
                                st.error("Selected account is not part of that match.")
                                break

                            winner_id = selected_account_id
                            loser_id = b["id"] if winner_id == a["id"] else a["id"]

                            # Apply "loser reset cost" for the stats game
                            apply_loss_to_loser(state, loser_id, rules_now)

                            m["winner_id"] = winner_id
                            m["loser_id"] = loser_id
                            save_state(tid, state)
                            st.success(f"Saved: winner #{winner_id}, loser #{loser_id} (loser incurred reset rule).")
                            st.rerun()

                cbtn1, cbtn2 = st.columns([2, 2])
                with cbtn1:
                    if st.button("Build next round", use_container_width=True):
                        state = build_next_round(state)
                        save_state(tid, state)
                        st.rerun()
                with cbtn2:
                    if st.button("Save state", type="secondary", use_container_width=True):
                        save_state(tid, state)
                        st.success("Saved.")
                        st.rerun()

            st.divider()

            # Export/Import state
            st.subheader("Export / Import")
            export_payload = {"rules": get_rules(), "tournament_id": tid, "state": state}
            st.download_button(
                "Download tournament JSON",
                data=json.dumps(export_payload, indent=2),
                file_name=f"tournament_{tid}.json",
                mime="application/json",
                use_container_width=True
            )

            upl = st.file_uploader("Import tournament JSON", type=["json"])
            if upl is not None:
                try:
                    payload = json.loads(upl.read().decode("utf-8"))
                    if "state" not in payload:
                        st.error("Invalid file.")
                    else:
                        save_state(tid, payload["state"])
                        st.success("Imported and saved.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
