#!/usr/bin/env python3
"""
STEP 7: read the group images from step 6 with a vision model and save clean events.

INPUT  (step 6), per video:   <step6>/<video_id>/group_000.png ... and manifest.json
       (step 3), per video:   <step3>/<video_id>/players.json   (optional, auto-detected)
                              {"1": "SHASHI BHAI!", "2": "KeNsEi>GoD", "3": "N7F NORMAL", "4": "LB PAVAN"}
OUTPUT, per video:            <out>/<video_id>/events.json       clean events (the file you use)
                              <out>/<video_id>/raw/group_000.json  model answer for each group (resume + audit)
                              <out>/all_videos_summary.json

Each group image has 8 rows. The model returns one simple object per row. This script
  - checks the answer (exactly 8 rows, valid types) and retries if it is wrong,
  - links every row to its event id through manifest.json,
  - ignores anything the model returns for padded (empty) slots,
  - fixes common mistakes, snaps names to the roster, checks times,
  - marks rows that need a human look with review = true and a list of flags.

USAGE
  python step7.py --step6 /path/step6 --out /path/step7
  python step7.py --step6 /path/step6 --out /path/step7 --step3 /path/step3
  python step7.py --step6 ... --out ... --videos 1 2 --workers 3
  python step7.py --step6 ... --out ... --force          (ignore saved answers and call the API again)

Needs OPENROUTER_API_KEY in the environment or a .env file.
"""
import argparse, base64, difflib, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================ configuration ============================
MODEL = os.getenv("VL_MODEL", "qwen/qwen3.7-flash")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0
MAX_TOKENS = 3000
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "2"))
NAME_CUTOFF = 0.75          # roster match score needed to accept a name

TYPES = ["landed", "got_eliminated", "eliminated_enemies", "revived",
         "triggered_team_revival", "unlocked_arsenal", "other", "unreadable", "empty"]

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["rows"],
    "properties": {"rows": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["row", "time", "type", "actor", "targets", "kills", "zone", "item", "text"],
        "properties": {
            "row":     {"type": "integer"},
            "time":    {"type": ["string", "null"]},
            "type":    {"type": "string", "enum": TYPES},
            "actor":   {"type": ["string", "null"]},
            "targets": {"type": "array", "items": {"type": "string"}},
            "kills":   {"type": ["integer", "null"]},
            "zone":    {"type": ["string", "null"]},
            "item":    {"type": ["string", "null"]},
            "text":    {"type": ["string", "null"]},
        }}}},
}

PROMPT = """You are reading a cropped Free Fire battle log. The image has 8 rows, top to bottom, numbered 0 to 7. Some rows at the bottom may be empty.
For each row return one object. Read the text exactly as shown. Do not guess.

Types:
- landed: "X landed"
- got_eliminated: "X was eliminated"
- eliminated_enemies: "X eliminated N enemy/enemies", optionally "in <zone>"
- revived: "X revived <player or players> with a Revival Card" (or Super Revival Card)
- triggered_team_revival: "X triggered Team Revival"
- unlocked_arsenal: "X unlocked an arsenal"
- other: any line that matches none of these (copy the text exactly)
- unreadable: a row with an event that you cannot read
- empty: a row with no event

Fields:
- time: the match time as MM:SS
- actor: the player who did the action
- targets: the players it happened to. For got_eliminated the eliminated player goes in targets and actor is null.
  For revived, list every revived player. Write the name if a name is shown, or just the number (for example "4") if only a coloured number square is shown.
- kills: the number N for eliminated_enemies
- zone: the place name after "in", if shown
- item: the card name for revived (Revival Card or Super Revival Card)
- text: the full line without the icon

Use null for anything that is not shown. Return exactly 8 rows with row = 0 to 7 as a JSON response in this format: {"rows": [...]}"""
# =======================================================================

TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")


class PermanentError(Exception):
    """An error that retrying cannot fix (bad key, rejected schema, ...)."""


# ------------------------------ API call ------------------------------
def send_request(client, api_key, body):
    """One HTTP call. Returns (status_code, parsed_json_or_text). Tests replace this function."""
    r = client.post(API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def build_body(image_b64, model, max_tokens):
    # Check if model supports structured outputs (OpenAI, Google, Anthropic, Fireworks)
    supports_structured = any(prefix in model.lower() for prefix in ["openai/", "google/", "anthropic/", "fireworks/"])
    
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    
    if supports_structured:
        # Use strict JSON schema for models that support it
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "battle_log_rows", "strict": True, "schema": SCHEMA}
        }
        body["provider"] = {"require_parameters": True}
    else:
        # Use basic JSON mode for other models (like Qwen)
        body["response_format"] = {"type": "json_object"}
    
    return body


def check_rows(parsed):
    """Return (rows, flags, error). Needs exactly 8 rows with valid types."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("rows"), list):
        return None, [], "answer has no 'rows' list"
    rows = parsed["rows"]
    if len(rows) != 8:
        return None, [], f"expected 8 rows, got {len(rows)}"
    flags = []
    for r in rows:
        if not isinstance(r, dict) or r.get("type") not in TYPES:
            return None, [], f"bad row or type: {str(r)[:80]}"
    try:
        idx = [int(r.get("row")) for r in rows]
    except (TypeError, ValueError):
        idx = []
    if sorted(idx) != list(range(8)):
        # the model miscounted: trust the order of the list instead
        for i, r in enumerate(rows):
            r["row"] = i
        flags.append("row_numbers_reassigned")
    else:
        for r, i in zip(rows, idx):
            r["row"] = i
    return rows, flags, None


def extract_group(client, api_key, image_path, model, max_attempts, max_tokens):
    """Call the model for one group image. Returns (rows, flags, raw_info)."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")
    last = "no attempt"
    tokens = max_tokens
    for attempt in range(1, max_attempts + 1):
        wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1)) * (0.5 + random.random())
        try:
            status, data = send_request(client, api_key, build_body(image_b64, model, tokens))
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = f"network error: {e}"
            time.sleep(wait); continue
        if status in (429, 500, 502, 503, 504):
            last = f"API {status}"
            time.sleep(wait); continue
        if status != 200:
            raise PermanentError(f"API {status}: {str(data)[:300]}")
        try:
            choice = data["choices"][0]
            msg = choice["message"].get("content") or ""
            if isinstance(msg, list):
                msg = "".join(p.get("text", "") for p in msg if isinstance(p, dict))
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            last = "unexpected API answer shape"
            time.sleep(wait); continue
        if finish == "length":
            tokens = int(tokens * 1.5)          # the answer was cut off: allow more tokens
            last = f"answer cut off (finish_reason=length), retrying with max_tokens={tokens}"
            continue
        clean_msg = msg.strip()
        if clean_msg.startswith("```"):
            clean_msg = re.sub(r"^```(?:json)?\s*", "", clean_msg)
            clean_msg = re.sub(r"\s*```$", "", clean_msg)
        try:
            parsed = json.loads(clean_msg)
        except json.JSONDecodeError as e:
            last = f"invalid JSON: {e}"
            time.sleep(wait); continue
        rows, flags, err = check_rows(parsed)
        if err:
            last = err
            continue
        return rows, flags, {"attempts": attempt, "usage": data.get("usage"), "finish_reason": finish}
    raise RuntimeError(f"gave up after {max_attempts} attempts: {last}")


# ------------------------- names, times, cleaning -------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def snap_name(name, roster_names):
    """Return (name, score). Accepts partial names such as 'PAVAN' for 'LB PAVAN'."""
    if not roster_names or not name:
        return name, None
    n = norm(name)
    best_score, best = 0.0, None
    for r in roster_names:
        rn = norm(r)
        score = difflib.SequenceMatcher(None, n, rn).ratio()
        if len(n) >= 3 and rn and (n in rn or rn in n):
            score = max(score, 0.85)
        if score > best_score:
            best_score, best = score, r
    if best_score >= NAME_CUTOFF:
        return best, best_score
    return name, best_score


def load_roster(roster_dir, video_id):
    """Return {slot_string: name}. Empty dict if there is no roster file.
    
    Looks for roster in two places:
    1. {roster_dir}/{video_id}.json (legacy format)
    2. {roster_dir}/{video_id}/players.json (step3 pipeline format)
    """
    if not roster_dir:
        return {}
    
    # Try step3 pipeline format first: {roster_dir}/{video_id}/players.json
    step3_path = os.path.join(roster_dir, str(video_id), "players.json")
    if os.path.exists(step3_path):
        path = step3_path
    else:
        # Fall back to legacy format: {roster_dir}/{video_id}.json
        path = os.path.join(roster_dir, f"{video_id}.json")
        if not os.path.exists(path):
            return {}
    
    data = json.load(open(path))
    if isinstance(data, dict) and "players" in data:
        data = data["players"]
    if isinstance(data, list):
        return {str(p["slot"]): p["name"] for p in data if "slot" in p and "name" in p}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def clean_text(v):
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def to_seconds(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def clean_row(r, roster):
    """Turn one model row into a clean event dict (without event_id and source)."""
    flags = []
    typ = r["type"]
    time_s = clean_text(r.get("time"))
    actor = clean_text(r.get("actor"))
    targets_raw = r.get("targets") or []
    # Guard: model sometimes returns a plain string instead of a list (schema non-compliance)
    if isinstance(targets_raw, str):
        targets_raw = [targets_raw]
    targets = [t for t in (clean_text(x) for x in targets_raw) if t]
    kills, zone, item = r.get("kills"), clean_text(r.get("zone")), clean_text(r.get("item"))
    text = clean_text(r.get("text"))

    if typ == "empty":                      # the manifest says this slot holds a real event
        typ = "unreadable"
        flags.append("model_said_empty")

    # common mistake: the eliminated player written as actor
    if typ == "got_eliminated" and actor and not targets:
        targets, actor = [actor], None
        flags.append("role_fixed")

    # fields that do not belong to this type
    if typ != "eliminated_enemies" and (kills is not None or zone):
        kills, zone = None, None
        flags.append("extra_fields_removed")
    if typ != "revived" and item:
        item = None
        flags.append("extra_fields_removed")

    # time
    if time_s and TIME_RE.match(time_s):
        mm, ss = time_s.split(":")
        time_s = f"{int(mm):02d}:{ss}"
    elif typ not in ("other", "unreadable"):
        flags.append("bad_time")
        time_s = time_s if time_s else None

    # names
    names = list(roster.values())
    if actor:
        actor, score = snap_name(actor, names)
        if names and (score is None or score < NAME_CUTOFF):
            flags.append("name_not_in_roster")
    resolved = []
    for t in targets:
        if re.fullmatch(r"\d", t):
            if roster:
                if t in roster:
                    resolved.append(roster[t])
                else:
                    resolved.append(t); flags.append("unknown_slot")
            else:
                resolved.append(t)           # no roster: keep the number
        else:
            t2, score = snap_name(t, names)
            if names and (score is None or score < NAME_CUTOFF):
                flags.append("name_not_in_roster")
            resolved.append(t2)
    targets = resolved

    # required fields per type
    need = {"landed": ["actor"], "triggered_team_revival": ["actor"], "unlocked_arsenal": ["actor"],
            "got_eliminated": ["targets"], "eliminated_enemies": ["actor", "kills"],
            "revived": ["actor", "targets"]}
    vals = {"actor": actor, "targets": targets, "kills": kills}
    for f in need.get(typ, []):
        if vals[f] in (None, [], ""):
            flags.append(f"missing_{f}")

    flags = sorted(set(flags))
    return {"time": time_s, "type": typ, "actor": actor, "targets": targets, "kills": kills,
            "zone": zone, "item": item, "text": text, "flags": flags,
            "review": bool(flags) or typ in ("other", "unreadable")}


# ------------------------------ one video ------------------------------
def process_video(video_id, vdir, out_dir, api_key, model, roster_dir, workers,
                  max_attempts, max_tokens, force):
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    manifest = json.load(open(os.path.join(vdir, "manifest.json")))
    roster = load_roster(roster_dir, video_id)
    print(f"\nvideo {video_id}: {len(manifest)} group images, roster {'found' if roster else 'NOT found'}")

    results, failed = {}, []
    tokens_in = tokens_out = 0

    def work(group):
        name = group["image"]
        raw_path = os.path.join(out_dir, "raw", name.replace(".png", ".json"))
        if os.path.exists(raw_path) and not force:
            saved = json.load(open(raw_path))
            return name, saved["rows"], saved.get("flags", []), saved.get("info", {}), True
        client = httpx.Client(timeout=90.0)
        try:
            rows, flags, info = extract_group(client, api_key, os.path.join(vdir, name),
                                              model, max_attempts, max_tokens)
        finally:
            client.close()
        json.dump({"model": model, "rows": rows, "flags": flags, "info": info},
                  open(raw_path, "w"), indent=1)
        return name, rows, flags, info, False

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(work, g): g["image"] for g in manifest}
        for fut in as_completed(futs):
            img = futs[fut]
            try:
                name, rows, flags, info, cached = fut.result()
                results[name] = (rows, flags)
                u = (info or {}).get("usage") or {}
                tokens_in += u.get("prompt_tokens", 0) or 0
                tokens_out += u.get("completion_tokens", 0) or 0
                print(f"  {name}: ok{' (saved answer)' if cached else ''}")
            except Exception as e:
                failed.append({"image": img, "error": str(e)})
                print(f"  {img}: FAILED: {e}")

    # ---- link rows to events through the manifest ----
    events, padding_ignored = [], 0
    for g in manifest:
        if g["image"] not in results:
            continue
        rows, group_flags = results[g["image"]]
        by_row = {r["row"]: r for r in rows}
        for slot in g["slots"]:
            row = by_row.get(slot["slot"])
            if slot["event_id"] is None:                 # padded slot: never an event
                if row and row["type"] != "empty":
                    padding_ignored += 1
                continue
            if row is None:
                row = {"row": slot["slot"], "time": None, "type": "unreadable", "actor": None,
                       "targets": [], "kills": None, "zone": None, "item": None, "text": None}
            ev = clean_row(row, roster)
            ev["flags"] = sorted(set(ev["flags"] + group_flags))
            ev["review"] = ev["review"] or bool(group_flags)
            ev["event_id"] = slot["event_id"]
            ev["source"] = f"{g['image']}#{slot['slot']}"
            events.append(ev)
    events.sort(key=lambda e: e["event_id"])

    # ---- match time should not go backwards from one event to the next ----
    prev = None
    for ev in events:
        if ev["time"] and TIME_RE.match(ev["time"]):
            sec = to_seconds(ev["time"])
            if prev is not None and sec < prev:
                ev["flags"] = sorted(set(ev["flags"] + ["time_out_of_order"]))
                ev["review"] = True
            else:
                prev = sec

    order = ["event_id", "time", "type", "actor", "targets", "kills", "zone", "item", "text",
             "review", "flags", "source"]
    events = [{k: e[k] for k in order} for e in events]
    by_type = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    summary = {"video_id": video_id, "model": model, "groups": len(manifest), "groups_failed": failed,
               "events": len(events), "by_type": by_type,
               "needs_review": sum(1 for e in events if e["review"]),
               "padding_rows_ignored": padding_ignored,
               "tokens_in": tokens_in, "tokens_out": tokens_out}
    with open(os.path.join(out_dir, "events.json"), "w") as f:
        json.dump({"video_id": video_id, "events": events}, f, indent=1, ensure_ascii=False)
    print(f"  saved {len(events)} events ({summary['needs_review']} need review), "
          f"{padding_ignored} padded rows ignored, {len(failed)} groups failed")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Step 7: extract battle log events with a vision model")
    ap.add_argument("--step6", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--step3", default=None, help="Path to step3 output for rosters (auto-detected if not provided)")
    ap.add_argument("--roster-dir", default=None, help="DEPRECATED: use --step3 instead")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set (environment or .env file)")
        return 1
    vids = a.videos or sorted(d for d in os.listdir(a.step6)
                              if os.path.isfile(os.path.join(a.step6, d, "manifest.json")))
    if not vids:
        print("ERROR: no videos with a manifest.json found")
        return 1
    
    # Auto-detect step3 directory from step6 path for roster loading
    roster_source = a.step3 or a.roster_dir
    if not roster_source:
        # Try to derive step3 path from step6 path (e.g., .../step6 -> .../step3)
        step6_parent = os.path.dirname(a.step6)
        potential_step3 = os.path.join(step6_parent, "step3")
        if os.path.isdir(potential_step3):
            roster_source = potential_step3
            print(f"Auto-detected rosters from: {roster_source}")
    
    print(f"STEP 7: {len(vids)} videos, model {a.model}")
    os.makedirs(a.out, exist_ok=True)
    allsum, bad = {}, 0
    for v in vids:
        try:
            s = process_video(v, os.path.join(a.step6, v), os.path.join(a.out, v), api_key, a.model,
                              roster_source, a.workers, a.max_attempts, a.max_tokens, a.force)
            allsum[v] = s
            bad += len(s["groups_failed"])
        except Exception as e:
            print(f"video {v}: ERROR {e}")
            allsum[v] = {"error": str(e)}
            bad += 1
    json.dump(allsum, open(os.path.join(a.out, "all_videos_summary.json"), "w"), indent=1)
    total = sum(s.get("events", 0) for s in allsum.values())
    print(f"\nDONE: {total} events in {len(vids)} videos"
          + (f", {bad} problems (see the lines above; rerun to retry failed groups)" if bad else ""))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())