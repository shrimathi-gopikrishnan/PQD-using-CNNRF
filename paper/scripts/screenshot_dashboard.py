#!/usr/bin/env python3
"""
Capture dashboard screenshots for the IEEE paper figures.

Run AFTER starting the backend (run.bat) so the dashboard is live at
http://localhost:5000/dashboard. The python_sender feeds the backend with
a mix of all 17 classes; this script polls the dashboard's classification
card and writes a screenshot the first time each of {Normal, Abnormal}
states is observed.

Outputs (into paper/figures/):
  dashboard.png       -- full overview, taken first
  pred_normal.png     -- captured when class == Pure_Sinusoidal
  pred_abnormal.png   -- captured when class != Pure_Sinusoidal

Setup (one time):
  venv312\\Scripts\\python.exe -m pip install playwright
  venv312\\Scripts\\python.exe -m playwright install chromium

Usage:
  venv312\\Scripts\\python.exe paper\\scripts\\screenshot_dashboard.py
  venv312\\Scripts\\python.exe paper\\scripts\\screenshot_dashboard.py --url http://localhost:5000/dashboard

Notes:
  * The script DOES NOT start the backend or the sender for you. Start them
    first via `run.bat` (or by hand: backend/app.py + tools/python_sender.py).
  * If the backend is not reachable the dashboard auto-falls-back to demo
    mode after ~3 s; the script will still capture screenshots from that.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "playwright is not installed. Run:\n"
        "  pip install playwright\n"
        "  playwright install chromium\n"
    )
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_class_label(page) -> str:
    """
    Return the currently displayed class name from the dashboard's
    classification card. Tries a small set of plausible selectors and
    falls back to the page innerText so the script keeps working if
    the dashboard markup shifts slightly.
    """
    candidates = [
        "[data-test='class-name']",
        "#class-name",
        ".class-name",
        ".classification-card .class",
        ".prediction h2",
        ".pred-class",
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                txt = (loc.first.inner_text() or "").strip()
                if txt:
                    return txt
            except Exception:
                continue
    try:
        return (page.evaluate("() => document.body.innerText") or "").strip()
    except Exception:
        return ""


def is_normal_label(label: str) -> bool:
    L = label.lower()
    if "abnormal" in L:
        return False
    return ("pure_sin" in L) or ("pure sinusoidal" in L) or (L == "normal")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:5000/dashboard",
                   help="dashboard URL (default: %(default)s)")
    p.add_argument("--width", type=int, default=1600,
                   help="viewport width (default: %(default)s)")
    p.add_argument("--height", type=int, default=1000,
                   help="viewport height (default: %(default)s)")
    p.add_argument("--scale", type=int, default=2,
                   help="device pixel ratio for retina-quality output "
                        "(default: %(default)s)")
    p.add_argument("--warmup-s", type=int, default=8,
                   help="seconds to let the dashboard connect / demo-mode "
                        "warm up before capturing (default: %(default)s)")
    p.add_argument("--poll-timeout-s", type=int, default=60,
                   help="how long to keep waiting for both Normal and "
                        "Abnormal states (default: %(default)s)")
    p.add_argument("--poll-interval-s", type=float, default=0.4,
                   help="how often to check the classification card "
                        "(default: %(default)s)")
    p.add_argument("--full-page", action="store_true", default=True,
                   help="capture the full scrollable page (default true)")
    p.add_argument("--no-full-page", dest="full_page", action="store_false")
    args = p.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=args.scale,
        )
        page = ctx.new_page()

        print(f"[1/4] opening {args.url}")
        try:
            page.goto(args.url, wait_until="networkidle", timeout=20_000)
        except Exception as e:
            print(f"      WARN: navigation reported {e!r}; "
                  f"continuing anyway")

        print(f"[2/4] warming up for {args.warmup_s}s "
              f"(dashboard may switch to demo mode if backend is offline)")
        time.sleep(args.warmup_s)

        # 1) Full dashboard overview
        out = FIG_DIR / "dashboard.png"
        page.screenshot(path=str(out), full_page=args.full_page)
        print(f"      wrote {out}")

        # 2) Wait for a Normal state and capture
        normal_done = False
        abnormal_done = False
        deadline = time.time() + args.poll_timeout_s
        last_label = ""
        print(f"[3/4] watching for Normal and Abnormal states "
              f"(timeout {args.poll_timeout_s}s)")
        while time.time() < deadline and not (normal_done and abnormal_done):
            label = read_class_label(page)
            if label != last_label:
                last_label = label
                short = label.replace("\n", " | ")[:80]
                print(f"      observed: {short!r}")
            if not label:
                time.sleep(args.poll_interval_s)
                continue
            if is_normal_label(label) and not normal_done:
                out = FIG_DIR / "pred_normal.png"
                page.screenshot(path=str(out), full_page=args.full_page)
                print(f"      wrote {out}")
                normal_done = True
            elif (not is_normal_label(label)) and not abnormal_done:
                out = FIG_DIR / "pred_abnormal.png"
                page.screenshot(path=str(out), full_page=args.full_page)
                print(f"      wrote {out}")
                abnormal_done = True
            time.sleep(args.poll_interval_s)

        if not normal_done:
            print("      WARN: never observed a Normal state; "
                  "pred_normal.png not written")
        if not abnormal_done:
            print("      WARN: never observed an Abnormal state; "
                  "pred_abnormal.png not written")

        print("[4/4] done")
        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
