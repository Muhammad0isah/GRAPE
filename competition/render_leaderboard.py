import csv
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"

def read_rows():
  if not CSV_PATH.exists():
    return []
  with CSV_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if (r.get("team") or "").strip()]
  return rows

def main():
  rows = read_rows()
  def score_key(r):
    try:
      return float(r.get("score","-inf"))
    except:
      return float("-inf")
  def ts_key(r):
    try:
      ts = (r.get("timestamp_utc", "") or "").strip()
      dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
      if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
      return dt
    except:
      return datetime.fromtimestamp(0, tz=timezone.utc)

  rows.sort(key=lambda r: (score_key(r), ts_key(r)), reverse=True)

  lines = []
  lines.append("# Leaderboard\n")
  lines.append("This leaderboard is **auto-updated** when a submission PR is merged. ")
  lines.append("For interactive search and filters, enable GitHub Pages and open **/docs/leaderboard.html**.\n\n")

  lines.append("| Rank | Team | Model | Type | Score | Date (UTC) | Notes |\n")
  lines.append("|---:|---|---|---|---:|---|---|\n")
  rank = 1
  for i, r in enumerate(rows):
    if i > 0:
      prev_score = rows[i-1].get("score", "")
      cur_score = r.get("score", "")
      if cur_score != prev_score:
        rank = i + 1
    team = (r.get("team") or "").strip()
    model = (r.get("model") or "").strip()
    stype = (r.get("type") or "").strip()
    score = (r.get("score") or "").strip()
    ts = (r.get("timestamp_utc") or "").strip()
    notes = (r.get("notes") or "").strip()
    model_disp = f"`{model}`" if model else "-"
    type_disp = f"`{stype}`" if stype else "-"
    notes_disp = notes if notes else "-"
    lines.append(f"| {rank} | {team} | {model_disp} | {type_disp} | {score} | {ts} | {notes_disp} |\n")

  MD_PATH.write_text("".join(lines), encoding="utf-8")

if __name__ == "__main__":
  main()
