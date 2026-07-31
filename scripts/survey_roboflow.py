"""Survey Clash Royale datasets on Roboflow Universe by REAL annotation counts.

Roboflow project metadata lists declared class names, which is misleading:
several CR projects advertise 120-250 classes covering the current roster
while almost all of those classes have zero annotations. This prints the
declared-vs-annotated split so datasets are judged on labels, not labels'
names.

Needs ROBOFLOW_API_KEY (free, app.roboflow.com -> Settings -> API Keys);
put it in .env (gitignored).

Run:  set -a && source .env && set +a
      .venv/bin/python scripts/survey_roboflow.py
"""

import json
import os
import sys
import urllib.request

PROJECTS = [
    ("clash-royale-swwaz", "cr-ai"),
    ("minesbot", "clash-royale-bot"),
    ("dargox", "clash-royale-kozno"),
    ("stuff-m0fm7", "clash-ai-kimrx"),
    ("fafa-zoa5z", "torchroyale-enemies"),
    ("clash-royale-k9ajk", "clash-royale-troop-detection-9pgyy"),
    ("nejc-zavodnik", "clash-royale-troop-detection"),
]
MODERN = ("bush", "berserker", "boss", "machine", "goblinstein", "rune",
          "spirit-empress", "hero", "evo")


def main() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY (see docstring)")
    for workspace, project in PROJECTS:
        url = f"https://api.roboflow.com/{workspace}/{project}?api_key={key}"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.load(response)["project"]
        except Exception as exc:  # noqa: BLE001 - survey script, report and continue
            print(f"{workspace}/{project}: unreachable ({exc})")
            continue
        classes = data.get("classes", {})
        annotated = {k: v for k, v in classes.items() if v}
        modern = {k: v for k, v in annotated.items()
                  if any(m in k.lower() for m in MODERN)}
        print(f"\n{workspace}/{project}: {data.get('images')} images, "
              f"license {data.get('license')}")
        print(f"  classes declared {len(classes)} | actually annotated {len(annotated)}")
        print(f"  total instances {sum(annotated.values())}")
        if modern:
            top = sorted(modern.items(), key=lambda kv: -kv[1])[:8]
            print(f"  post-2024 units: {dict(top)}")
        else:
            print("  post-2024 units: none annotated")


if __name__ == "__main__":
    main()
