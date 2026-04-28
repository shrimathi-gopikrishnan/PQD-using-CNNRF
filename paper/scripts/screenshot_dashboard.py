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
    classification card (#dispName). Tries a few legacy selectors first
    so the script also works against older dashboard builds, but never
    falls back to whole-page innerText (the legend always contains
    "Pure_Sinusoidal", which would fool the normal/abnormal check).
    """
    candidates = [
        "#dispName",
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
                if txt and txt != "—":
                    return txt
            except Exception:
                continue
    return ""


def is_normal_label(label: str) -> bool:
    """Match only the exact pure-sinusoidal label, not arbitrary substrings."""
    L = label.strip().lower().replace(" ", "_")
    return L in {"pure_sinusoidal", "normal_-_pure_sinusoidal", "normal"}


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
    p.add_argument("--warmup-s", type=int, default=12,
                   help="seconds to let the dashboard connect / demo-mode "
                        "warm up before capturing (default: %(default)s)")
    p.add_argument("--poll-timeout-s", type=int, default=60,
                   help="how long to keep waiting for both Normal and "
                        "Abnormal states (default: %(default)s)")
    p.add_argument("--poll-interval-s", type=float, default=0.4,
                   help="how often to check the classification card "
                        "(default: %(default)s)")
    args = p.parse_args()

    # CSS injected into the dashboard before every screenshot.
    #   * background-attachment:fixed leaves white below the first viewport
    #     when Chromium stitches a full-page screenshot, so force scroll.
    #   * a solid dark fill on html/body matches the gradient's bottom stop
    #     so any pixel the gradient does not cover is still dark, not white.
    #   * hide scrollbars so they do not appear in the capture.
    SHOT_CSS = """
      html, body {
        background-attachment: scroll !important;
        background-color: #0d1118 !important;
      }
      ::-webkit-scrollbar { width: 0 !important; height: 0 !important; }
      * { scrollbar-width: none !important; }
      /* Hide the "Stream paused" toast that pops up while we freeze the
         stream to take a clean shot — it would otherwise show in the
         bottom-right of every captured image. */
      #toast, .toast { opacity: 0 !important; visibility: hidden !important; }
    """

    def prep_page(page) -> None:
        """One-shot setup: inject CSS to fix the gradient + scrollbars and
        resize the viewport to fit the real content height. Done once during
        warmup so the per-shot path can just snap without stalling — by the
        time the polling loop sees the desired class, the dashboard may have
        cycled to a different one if we were to delay capture by a second."""
        page.add_style_tag(content=SHOT_CSS)
        page.wait_for_timeout(500)
        height = page.evaluate(
            "() => Math.max("
            "document.documentElement.scrollHeight,"
            "document.body.scrollHeight)"
        )
        height = max(int(height) + 12, 600)
        height = min(height, 4000)
        page.set_viewport_size({"width": args.width, "height": height})
        page.wait_for_timeout(400)

    def is_paused(page) -> bool:
        try:
            return page.eval_on_selector(
                "#btnPause",
                "el => el.classList.contains('active')"
            )
        except Exception:
            return False

    def set_paused(page, want: bool) -> None:
        """Toggle the dashboard's pause button until its state matches `want`.
        The script's `state.paused` is closure-scoped (not on window), so we
        drive the UI button instead of poking JS state directly."""
        for _ in range(2):
            if is_paused(page) == want:
                return
            try:
                page.click("#btnPause", timeout=1000)
            except Exception:
                return
            page.wait_for_timeout(60)

    def shoot(page, out_path: Path) -> None:
        """Freeze the dashboard, snap, then resume. Without the freeze, the
        live stream cycles to the next class between the label-read and the
        screenshot, and the captured image shows a different disturbance
        than the one we were waiting for."""
        set_paused(page, True)
        page.wait_for_timeout(120)
        page.screenshot(path=str(out_path), full_page=False)
        set_paused(page, False)

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
        prep_page(page)

        # 1) Full dashboard overview
        out = FIG_DIR / "dashboard.png"
        shoot(page, out)
        print(f"      wrote {out}")

        # 2) Wait for a Normal state and capture.
        # Pause-first protocol: each poll, freeze the stream, read the
        # currently displayed class, decide, capture (already paused), then
        # resume. This guarantees the captured frame matches the label that
        # triggered the capture — without a freeze, Socket.IO frames overwrite
        # the DOM between the label read and the screenshot.
        normal_done = False
        abnormal_done = False
        deadline = time.time() + args.poll_timeout_s
        last_label = ""
        print(f"[3/4] watching for Normal and Abnormal states "
              f"(timeout {args.poll_timeout_s}s)")
        while time.time() < deadline and not (normal_done and abnormal_done):
            set_paused(page, True)
            page.wait_for_timeout(80)  # let in-flight render flush
            label = read_class_label(page)
            if label != last_label:
                last_label = label
                short = label.replace("\n", " | ")[:80]
                print(f"      observed: {short!r}")
            if not label:
                set_paused(page, False)
                time.sleep(args.poll_interval_s)
                continue
            captured = False
            if is_normal_label(label) and not normal_done:
                out = FIG_DIR / "pred_normal.png"
                page.screenshot(path=str(out), full_page=False)
                print(f"      wrote {out}")
                normal_done = True
                captured = True
            elif (not is_normal_label(label)) and not abnormal_done:
                out = FIG_DIR / "pred_abnormal.png"
                page.screenshot(path=str(out), full_page=False)
                print(f"      wrote {out}")
                abnormal_done = True
                captured = True
            set_paused(page, False)
            # if we captured, give the stream a moment to advance away from
            # the just-captured class so we don't re-trigger on the same one
            time.sleep(args.poll_interval_s if not captured else 0.6)

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
