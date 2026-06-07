"""Capture documentation screenshots from the running Streamlit app.

Usage (with the app running on http://localhost:8501):
    .venv/bin/python scripts/capture_screenshots.py

Writes PNGs into docs/screenshots/. Drives the key-free views (walkthrough,
talent-pool ranking, retrieval eval) so no API key is needed.
"""

from __future__ import annotations

import pathlib

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = pathlib.Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def _main_el(page):
    for sel in ('[data-testid="stMain"]', "section.main", '[data-testid="stAppViewContainer"]'):
        loc = page.locator(sel).first
        if loc.count():
            return loc
    return page


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        page.set_default_timeout(120_000)
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector('[data-testid="stAppViewContainer"]')
        page.wait_for_timeout(2000)

        def shot(name, fn):
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {name}: {e}")

        # 1. Hero — top of "How RAG works"
        def hero():
            page.wait_for_selector("text=How RAG works")
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / "01-hero.png"))
        shot("01-hero", hero)

        # 2. Hierarchical chunking expander
        def chunking():
            exp = page.locator('[data-testid="stExpander"]', has_text="CHUNK — hierarchical").first
            exp.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            exp.screenshot(path=str(OUT / "02-chunking.png"))
        shot("02-chunking", chunking)

        # 3. Retrieve pipeline — auto-computes from the default query (table + plot)
        def retrieve():
            exp = page.locator('[data-testid="stExpander"]', has_text="RETRIEVE — small-to-big").first
            exp.scroll_into_view_if_needed()
            page.wait_for_selector('[data-testid="stPlotlyChart"]')
            page.wait_for_timeout(2000)
            exp.screenshot(path=str(OUT / "03-retrieve.png"))
        shot("03-retrieve", retrieve)

        # 4. Talent pool — multi-résumé ranking
        def talent():
            page.get_by_text("🏢 Talent pool", exact=True).first.click()
            page.wait_for_selector("text=Pool ready:")
            page.wait_for_timeout(800)
            page.get_by_role("button", name="Search the pool").click()
            page.wait_for_selector("text=ranked (filtered")
            page.wait_for_timeout(1500)
            # Streamlit scrolls inside a fixed-height container, so grow the
            # viewport tall enough that all ranked candidates render at once.
            page.set_viewport_size({"width": 1440, "height": 2600})
            page.wait_for_timeout(800)
            _main_el(page).screenshot(path=str(OUT / "04-talent-pool.png"))
            page.set_viewport_size({"width": 1440, "height": 1000})
        shot("04-talent-pool", talent)

        # 5. Retrieval evaluation chart
        def evaluation():
            page.get_by_text("📊 Retrieval evaluation", exact=True).first.click()
            page.wait_for_timeout(500)
            page.get_by_role("button", name="Run retrieval eval").click()
            page.wait_for_selector('[data-testid="stPlotlyChart"]')
            page.wait_for_timeout(2000)
            _main_el(page).screenshot(path=str(OUT / "05-eval.png"))
        shot("05-eval", evaluation)

        browser.close()
        print("Done. Screenshots in", OUT)


if __name__ == "__main__":
    run()
