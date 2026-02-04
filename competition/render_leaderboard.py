import csv
from pathlib import Path
from datetime import datetime

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
      return datetime.fromisoformat(r.get("timestamp_utc","").replace("Z","+00:00"))
    except:
      return datetime.fromtimestamp(0)

  rows.sort(key=lambda r: (score_key(r), ts_key(r)), reverse=True)

  lines = []
  lines.append("# Leaderboard\n")
  lines.append("This leaderboard is **auto-updated** when a submission PR is merged. ")
  lines.append("For interactive search and filters, enable GitHub Pages and open **/docs/leaderboard.html**.\n\n")

  lines.append("| Rank | Team | Model | Score | Date (UTC) | Notes |\n")
  lines.append("|---:|---|---|---:|---|---|\n")
  for i, r in enumerate(rows, start=1):
    team = (r.get("team") or "").strip()
    model = (r.get("model") or "").strip()
    score = (r.get("score") or "").strip()
    ts = (r.get("timestamp_utc") or "").strip()
    notes = (r.get("notes") or "").strip()
    model_disp = f"`{model}`" if model else ""
    lines.append(f"| {i} | {team} | {model_disp} | {score} | {ts} | {notes} |\n")

  MD_PATH.write_text("".join(lines), encoding="utf-8")

if __name__ == "__main__":
  main()
