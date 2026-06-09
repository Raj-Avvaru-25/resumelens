"""Shared UI: the ResumeLens visual system (theme, hero, landing, helpers)."""

from __future__ import annotations

import streamlit as st
from streamlit.components.v1 import html as _html

from rag import config

# A premium, decelerating easing curve (easeOutExpo-ish) used across the app.
_EASE = "cubic-bezier(.16, 1, .3, 1)"

# Subtle film grain for an editorial, printed feel.
_GRAIN = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence "
    "type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E"
    "%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' "
    "opacity='0.55'/%3E%3C/svg%3E\")"
)


def inject_global_styles() -> None:
    """Apply the shared ResumeLens visual system + the living background layer."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        :root {{
          --rl-bg: #f3f1ec;
          --rl-surface: rgba(255, 255, 255, 0.82);
          --rl-surface-solid: #ffffff;
          --rl-text: #171714;
          --rl-muted: #68707d;
          --rl-line: rgba(23, 23, 20, .12);
          --rl-accent: #5746d8;
          --rl-accent-2: #7c5cf0;
          --rl-accent-soft: #ece9ff;
          --rl-mint: #dff8ef;
          --rl-blue: #e8f1ff;
          --rl-radius: 18px;
          --rl-shadow: 0 12px 34px rgba(34, 37, 53, 0.06);
          --rl-ease: {_EASE};
        }}

        html, body, [class*="css"] {{
          font-family: "Inter", sans-serif;
          scroll-behavior: smooth;
        }}

        body {{ background: var(--rl-bg); }}
        .stApp {{ color: var(--rl-text); background: transparent; }}

        /* ---- Living background (slow drifting glows + film grain) ---------- */
        .rl-bg {{
          position: fixed;
          inset: 0;
          z-index: -2;
          overflow: hidden;
          background: var(--rl-bg);
        }}
        .rl-bg span {{
          position: absolute;
          border-radius: 50%;
          filter: blur(98px);
          will-change: transform;
        }}
        .rl-bg .g1 {{
          width: 52vw; height: 52vw; left: -18vw; top: -24vw;
          background: radial-gradient(circle, rgba(124, 92, 240, .11), transparent 64%);
          animation: rl-drift1 34s var(--rl-ease) infinite alternate;
        }}
        .rl-bg .g2 {{
          width: 46vw; height: 46vw; right: -16vw; top: 34vw;
          background: radial-gradient(circle, rgba(64, 196, 160, .08), transparent 64%);
          animation: rl-drift2 40s var(--rl-ease) infinite alternate;
        }}
        .rl-bg .g3 {{
          width: 44vw; height: 44vw; left: 34vw; bottom: -28vw;
          background: radial-gradient(circle, rgba(86, 142, 255, .07), transparent 64%);
          animation: rl-drift3 46s var(--rl-ease) infinite alternate;
        }}
        .rl-bg::after {{
          content: "";
          position: absolute;
          inset: 0;
          opacity: .03;
          background-image: {_GRAIN};
        }}
        @keyframes rl-drift1 {{ to {{ transform: translate(7vw, 5vw) scale(1.18); }} }}
        @keyframes rl-drift2 {{ to {{ transform: translate(-6vw, -4vw) scale(1.14); }} }}
        @keyframes rl-drift3 {{ to {{ transform: translate(4vw, -6vw) scale(1.2); }} }}

        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stDecoration"], footer {{ visibility: hidden; height: 0; }}
        [data-testid="stToolbar"] {{ background: transparent; }}

        /* ---- Sidebar collapse "Menu" control ------------------------------ */
        [data-testid="stSidebarCollapsedControl"] {{
          position: fixed; top: 1rem; left: 1rem; z-index: 999999;
          visibility: visible !important; opacity: 1 !important;
        }}
        [data-testid="stSidebarCollapsedControl"] button {{
          position: relative; width: 6.7rem; height: 2.75rem;
          padding: 0 .9rem 0 2.6rem; color: var(--rl-text);
          background: rgba(255,255,255,.94); border: 1px solid var(--rl-line);
          border-radius: 12px; box-shadow: 0 8px 24px rgba(34,31,25,.14);
          backdrop-filter: blur(14px);
          transition: transform .25s var(--rl-ease), box-shadow .25s var(--rl-ease), background-color .25s var(--rl-ease);
        }}
        [data-testid="stSidebarCollapsedControl"] button::before {{
          content: ""; position: absolute; left: .9rem; top: 50%;
          width: 1.05rem; height: .72rem; transform: translateY(-50%);
          background:
            linear-gradient(var(--rl-text), var(--rl-text)) 0 0 / 100% 2px no-repeat,
            linear-gradient(var(--rl-text), var(--rl-text)) 0 50% / 100% 2px no-repeat,
            linear-gradient(var(--rl-text), var(--rl-text)) 0 100% / 100% 2px no-repeat;
          opacity: .8;
        }}
        [data-testid="stSidebarCollapsedControl"] button::after {{
          content: "Menu"; color: var(--rl-text);
          font: 700 .82rem "Inter", sans-serif; letter-spacing: -.01em;
        }}
        [data-testid="stSidebarCollapsedControl"] button svg {{ display: none; }}
        [data-testid="stSidebarCollapsedControl"] button:hover {{
          color: var(--rl-accent); background: white;
          transform: translateY(-2px); box-shadow: 0 12px 30px rgba(34,31,25,.18);
        }}
        [data-testid="stSidebarCollapsedControl"] button:hover::before {{
          background:
            linear-gradient(var(--rl-accent), var(--rl-accent)) 0 0 / 100% 2px no-repeat,
            linear-gradient(var(--rl-accent), var(--rl-accent)) 0 50% / 100% 2px no-repeat,
            linear-gradient(var(--rl-accent), var(--rl-accent)) 0 100% / 100% 2px no-repeat;
        }}
        [data-testid="stSidebarCollapsedControl"] button:hover::after {{ color: var(--rl-accent); }}

        [data-testid="stMainBlockContainer"] {{
          max-width: 1320px;
          padding: 1.6rem 2.5rem 4rem;
          animation: rl-page-in .55s var(--rl-ease) both;
        }}

        /* ---- Sidebar ------------------------------------------------------- */
        [data-testid="stSidebar"] {{
          background: rgba(245, 243, 238, .95);
          border-right: 1px solid rgba(23, 23, 20, .08);
          backdrop-filter: blur(14px);
        }}
        [data-testid="stSidebar"]::before {{
          content: "🔍 ResumeLens"; display: block;
          margin: .9rem 1.1rem .4rem; color: var(--rl-text);
          font: 800 1.1rem/1 "Manrope", sans-serif; letter-spacing: -.04em;
        }}
        /* Compact the sidebar so it fits without scrolling. */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .25rem; }}
        [data-testid="stSidebar"] hr {{ margin: .4rem 0; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{ padding: .3rem .6rem; }}
        [data-testid="stSidebarNav"] {{ padding-top: 0; margin-bottom: .2rem; }}
        [data-testid="stSidebarNav"] ul,
        [data-testid="stSidebarNavItems"] {{ gap: 0 !important; }}
        [data-testid="stSidebarNavItems"] li {{ margin: 0 !important; }}
        [data-testid="stSidebarNav"] a {{
          min-height: 1.85rem; margin: 0 8px; border-radius: 9px;
          transition: background-color .25s var(--rl-ease), color .25s var(--rl-ease), transform .25s var(--rl-ease);
        }}
        /* Nav section labels (Learn / Ask / Analyze) — tighten their spacing. */
        [data-testid="stSidebarNav"] [class*="navSectionHeader"],
        [data-testid="stSidebarNav"] li > div:not([data-testid]) {{
          margin: .35rem 0 .1rem !important;
        }}
        /* Explicit "✕ Close", always visible (not only on hover). */
        [data-testid="stSidebarCollapseButton"] {{ opacity: 1 !important; visibility: visible !important; }}
        [data-testid="stSidebarCollapseButton"] button {{
          width: auto !important; padding: 0 .75rem !important; height: 2rem !important;
          font-size: 0 !important;
          color: var(--rl-text) !important;
          border: 1px solid var(--rl-line) !important;
          border-radius: 8px !important;
          background: rgba(255,255,255,.85) !important;
          opacity: 1 !important;
        }}
        [data-testid="stSidebarCollapseButton"] button * {{ display: none !important; }}
        [data-testid="stSidebarCollapseButton"] button::after {{
          content: "✕ Close"; font: 700 .8rem "Inter", sans-serif; white-space: nowrap;
        }}
        [data-testid="stSidebarCollapseButton"] button:hover {{
          background: var(--rl-accent-soft) !important; color: var(--rl-accent) !important;
        }}
        [data-testid="stSidebarNav"] a:hover {{
          background: #f2f1ff; color: var(--rl-accent); transform: translateX(3px);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
          color: var(--rl-accent); background: var(--rl-accent-soft); font-weight: 700;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: var(--rl-muted); }}

        h1, h2, h3, h4 {{
          font-family: "Manrope", sans-serif !important;
          color: var(--rl-text) !important; letter-spacing: -.045em;
        }}
        /* Hide the auto-generated heading anchor (chain-link) icons. */
        [data-testid="stHeadingWithActionElements"] a {{ display: none !important; }}
        p, li, label, [data-testid="stCaptionContainer"] {{ color: var(--rl-muted); }}

        /* ---- Cards / surfaces --------------------------------------------- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
          background: var(--rl-surface);
          border: 1px solid var(--rl-line);
          border-radius: var(--rl-radius);
          box-shadow: var(--rl-shadow);
          backdrop-filter: blur(8px);
          transition: transform .5s var(--rl-ease), box-shadow .5s var(--rl-ease), border-color .5s var(--rl-ease);
          animation: rl-rise .7s var(--rl-ease) both;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
          transform: translateY(-4px);
          border-color: #d4d0fb;
          box-shadow: 0 22px 50px rgba(34, 37, 53, .12);
        }}

        [data-testid="stExpander"] {{
          background: rgba(255, 255, 255, .74);
          border: 1px solid var(--rl-line);
          border-radius: 14px; overflow: hidden;
          transition: border-color .3s var(--rl-ease), background-color .3s var(--rl-ease);
        }}
        [data-testid="stExpander"]:hover {{ border-color: #d7d3fa; background: rgba(255,255,255,.96); }}

        [data-testid="stMetric"] {{
          background: var(--rl-surface-solid); border: 1px solid var(--rl-line);
          border-radius: 14px; padding: .85rem 1rem; box-shadow: 0 8px 22px rgba(34,37,53,.04);
        }}
        [data-testid="stMetricValue"] {{
          color: var(--rl-text); font-family: "Manrope", sans-serif; letter-spacing: -.05em;
        }}

        /* ---- Buttons ------------------------------------------------------- */
        .stButton > button, .stDownloadButton > button {{
          min-height: 2.65rem; border: 1px solid var(--rl-line); border-radius: 11px;
          background: var(--rl-surface-solid); color: var(--rl-text); font-weight: 700;
          box-shadow: 0 3px 10px rgba(34,37,53,.04);
          transition: transform .28s var(--rl-ease), box-shadow .28s var(--rl-ease), border-color .28s var(--rl-ease), background-color .28s var(--rl-ease), color .28s var(--rl-ease);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
          transform: translateY(-2px); border-color: #c8c3f7; color: var(--rl-accent);
          box-shadow: 0 10px 22px rgba(75, 64, 180, .14);
        }}
        .stButton > button:active, .stDownloadButton > button:active {{
          transform: translateY(0) scale(.985); box-shadow: none;
        }}
        .stButton > button[kind="primary"] {{
          color: white; border-color: transparent;
          background: linear-gradient(120deg, var(--rl-accent), var(--rl-accent-2));
          box-shadow: 0 10px 22px rgba(102, 87, 232, .28);
        }}
        .stButton > button[kind="primary"]:hover {{
          color: white; filter: brightness(1.05);
          box-shadow: 0 14px 30px rgba(102, 87, 232, .34);
        }}
        .stButton > button p, .stDownloadButton > button p {{ color: inherit !important; }}

        /* ---- Inputs -------------------------------------------------------- */
        input, textarea, [data-baseweb="select"] > div {{
          color: var(--rl-text) !important; background: rgba(255,255,255,.92) !important;
          border-color: var(--rl-line) !important; border-radius: 11px !important;
          transition: border-color .25s var(--rl-ease), box-shadow .25s var(--rl-ease) !important;
        }}
        input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within {{
          border-color: #aaa2f2 !important; box-shadow: 0 0 0 4px rgba(102,87,232,.10) !important;
        }}

        [data-testid="stChatMessage"] {{
          padding: .8rem 1rem; margin: .65rem 0; background: rgba(255,255,255,.78);
          border: 1px solid var(--rl-line); border-radius: 15px;
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
          margin-left: 8%; background: var(--rl-accent-soft); border-color: #ddd9ff;
        }}
        [data-testid="stAlert"] {{ border: 1px solid var(--rl-line); border-radius: 13px; background: rgba(255,255,255,.76); }}
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{
          overflow: hidden; background: var(--rl-surface-solid);
          border: 1px solid var(--rl-line); border-radius: 14px;
        }}
        code, pre, .stCode {{ border-radius: 12px !important; }}
        hr {{ border-color: var(--rl-line) !important; }}

        @keyframes rl-rise {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes rl-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes rl-page-in {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes rl-image-in {{ from {{ opacity: .55; transform: scale(1.03); }} to {{ opacity: 1; transform: scale(1); }} }}

        /* ---- Hero (home) — compact so the landing fits one screen --------- */
        .rl-hero {{
          position: relative; overflow: hidden;
          min-height: min(46vh, 460px);
          display: flex; align-items: flex-end;
          padding: clamp(1.4rem, 3.5vw, 2.8rem);
          margin-bottom: 1rem; color: white; background: #292722;
          border-radius: 24px; box-shadow: 0 30px 70px rgba(34, 31, 25, .20);
          isolation: isolate; animation: rl-rise .75s var(--rl-ease) both;
        }}
        .rl-hero::before {{
          content: ""; position: absolute; inset: 0; z-index: -2;
          background-image: var(--hero-image); background-size: cover; background-position: center;
          animation: rl-image-in .9s var(--rl-ease) both;
          transition: transform 1.1s var(--rl-ease), filter 1.1s ease;
        }}
        .rl-hero::after {{
          content: ""; position: absolute; inset: 0; z-index: -1;
          background:
            linear-gradient(90deg, rgba(15,14,12,.78), rgba(15,14,12,.22) 64%, rgba(15,14,12,.08)),
            linear-gradient(0deg, rgba(15,14,12,.5), transparent 58%);
        }}
        .rl-hero:hover::before {{ transform: scale(1.03); filter: saturate(1.08); }}
        .rl-hero-copy {{ position: relative; z-index: 1; max-width: 760px; }}
        .rl-kicker, .rl-eyebrow {{
          color: var(--rl-accent); font-size: .72rem; font-weight: 800;
          letter-spacing: .09em; text-transform: uppercase;
        }}
        .rl-hero h1 {{
          position: relative; z-index: 1; max-width: 760px; margin: .5rem 0 .7rem;
          color: white !important; font-size: clamp(2.4rem, 5vw, 4.4rem);
          line-height: 1.05; letter-spacing: -.05em;
          animation: rl-rise .8s var(--rl-ease) both; animation-delay: .06s;
        }}
        .rl-hero p {{
          max-width: 600px; margin: 0; color: rgba(255,255,255,.84);
          font-size: 1.02rem; line-height: 1.6;
          animation: rl-rise .9s var(--rl-ease) both; animation-delay: .16s;
        }}
        .rl-hero .rl-kicker {{ color: rgba(255,255,255,.8); }}

        /* ---- Editorial image (recruiter banner) --------------------------- */
        .rl-editorial-image {{
          position: relative; overflow: hidden; min-height: 290px;
          margin: .25rem 0 1.4rem; border-radius: 20px;
          background-image: var(--editorial-image); background-size: 104%; background-position: center 56%;
          box-shadow: 0 22px 55px rgba(34,31,25,.15);
          animation: rl-rise .7s var(--rl-ease) both;
          transition: background-size 1.2s var(--rl-ease); isolation: isolate;
        }}
        .rl-editorial-image:hover {{ background-size: 109%; }}
        .rl-editorial-image::after {{
          content: ""; position: absolute; inset: 0; z-index: 0;
          background: linear-gradient(90deg, rgba(18,16,13,.62), transparent 66%);
        }}
        .rl-editorial-image-copy {{ position: relative; z-index: 1; max-width: 430px; padding: 1.5rem; color: white; }}
        .rl-editorial-image-copy h3 {{ margin: .3rem 0 .45rem; color: white !important; font-size: clamp(1.8rem, 4vw, 3rem); line-height: 1; }}
        .rl-editorial-image-copy p {{ color: rgba(255,255,255,.82); }}

        .rl-section-intro {{ display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin: 2rem 0 .8rem; }}
        .rl-section-intro h2 {{ margin: 0; font-size: 1.65rem; }}
        .rl-section-intro p {{ max-width: 520px; margin: 0; text-align: right; font-size: .88rem; }}

        .rl-page-head {{ max-width: 820px; margin: .2rem 0 1.4rem; }}
        .rl-page-head h1 {{ margin: .3rem 0 .45rem; font-size: clamp(1.85rem, 3vw, 2.8rem); line-height: 1.05; letter-spacing: -.04em; overflow-wrap: anywhere; }}
        .rl-page-head p {{ margin: 0; font-size: .98rem; line-height: 1.6; }}

        .rl-chip {{
          display: inline-flex; padding: .3rem .55rem; color: var(--rl-accent);
          background: var(--rl-accent-soft); border-radius: 999px;
          font-size: .68rem; font-weight: 800; letter-spacing: .05em;
        }}

        .rl-sidebar-resume {{
          margin: .15rem 0 .75rem; padding: .75rem .8rem;
          background: var(--rl-accent-soft); border: 1px solid #ddd9ff; border-radius: 12px;
        }}
        .rl-sidebar-resume small {{
          display: block; margin-bottom: .2rem; color: var(--rl-accent);
          font-size: .67rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
        }}
        .rl-sidebar-resume strong {{
          display: block; overflow: hidden; color: var(--rl-text);
          font-size: .86rem; text-overflow: ellipsis; white-space: nowrap;
        }}

        @media (prefers-reduced-motion: reduce) {{
          *, *::before, *::after {{ animation: none !important; transition-duration: .01ms !important; }}
        }}

        @media (max-width: 900px) {{
          [data-testid="stSidebarCollapsedControl"] {{ top: .6rem; left: .6rem; }}
          [data-testid="stSidebarCollapsedControl"] button {{ width: 2.75rem; padding: 0; border-radius: 999px; }}
          [data-testid="stSidebarCollapsedControl"] button::before {{ left: 50%; transform: translate(-50%, -50%); }}
          [data-testid="stSidebarCollapsedControl"] button::after {{ display: none; }}
          .rl-hero {{ min-height: 420px; padding: 1.6rem; }}
          .rl-section-intro {{ display: block; }}
          .rl-section-intro p {{ margin-top: .35rem; text-align: left; }}
        }}
        </style>
        <div class="rl-bg"><span class="g1"></span><span class="g2"></span><span class="g3"></span></div>
        """,
        unsafe_allow_html=True,
    )


def inject_home_layout() -> None:
    """Immersive landing: a full-screen hero image with the workflow options laid
    over it as dark-glass cards. Full-bleed, compact, sidebar collapsed."""
    st.markdown(
        """
        <style>
        /* On the landing the sidebar never opens — you navigate via the cards. */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }

        /* Full-screen hero image behind the landing. A fixed, viewport-sized
           layer — works only because nothing on the page uses transform/filter
           (those would re-anchor it and clip it). The page motion is opacity-only. */
        .rl-home-bg {
          position: fixed; inset: 0; z-index: -1;
          background:
            linear-gradient(102deg, rgba(13,12,10,.9), rgba(13,12,10,.52) 48%, rgba(13,12,10,.22)),
            linear-gradient(0deg, rgba(13,12,10,.74), transparent 56%),
            url('/app/static/resumelens-hero.png') center / cover no-repeat;
          animation: rl-fade 1s var(--rl-ease) both;
        }
        [data-testid="stMainBlockContainer"] {
          max-width: none !important;
          padding: clamp(1rem, 4vh, 2.6rem) clamp(1.2rem, 4vw, 4rem) 2rem !important;
          min-height: 88vh;
        }

        /* Hero text overlay (no boxed image). */
        .rl-hero-overlay { max-width: 780px; margin: 1vh 0 3.6vh; }
        .rl-hero-overlay .rl-kicker { color: rgba(255,255,255,.82); }
        .rl-brandmark {
          display: inline-flex; align-items: center; gap: .5rem; color: #fff;
          font: 800 2.1rem/1 "Manrope", sans-serif; letter-spacing: -.04em; margin-bottom: .5rem;
        }
        .rl-brandmark span { font-size: 1.5rem; }
        .rl-hero-overlay h1 {
          margin: .4rem 0 .55rem; color: #fff !important;
          font-size: clamp(2.1rem, 4.4vw, 3.7rem); line-height: 1.04; letter-spacing: -.05em;
          text-shadow: 0 2px 36px rgba(0,0,0,.5);
          animation: rl-rise .8s var(--rl-ease) both; animation-delay: .05s;
        }
        .rl-hero-overlay p {
          max-width: 580px; color: rgba(255,255,255,.88); font-size: .98rem; line-height: 1.55;
          animation: rl-rise .8s var(--rl-ease) both; animation-delay: .12s;
        }

        /* All landing text sits over a dark photo → force it light. */
        [data-testid="stMainBlockContainer"] h1,
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3,
        [data-testid="stMainBlockContainer"] h4,
        [data-testid="stMainBlockContainer"] p,
        [data-testid="stMainBlockContainer"] li,
        [data-testid="stMainBlockContainer"] label,
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] {
          color: rgba(255,255,255,.88) !important;
          text-shadow: 0 1px 10px rgba(0,0,0,.5);
        }
        [data-testid="stMainBlockContainer"] h4 {
          color: #fff !important; font-weight: 800; text-shadow: 0 2px 14px rgba(0,0,0,.55);
        }

        /* Cards: dark glass so text stays crisp over the photo. st.container(
           border=True, key="rlcard_*") adds a stable .st-key-rlcard_* class. */
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard"] {
          background: rgba(12,10,9,.07) !important;
          border: 1px solid rgba(255,255,255,.22) !important;
          border-radius: 16px !important;
          -webkit-backdrop-filter: blur(3px) saturate(1.15) !important;
          backdrop-filter: blur(3px) saturate(1.15) !important;
          box-shadow: 0 22px 55px rgba(0,0,0,.45) !important;
          transition: border-color .4s var(--rl-ease), box-shadow .4s var(--rl-ease), transform .4s var(--rl-ease) !important;
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard"]:hover {
          border-color: rgba(255,255,255,.34) !important;
          box-shadow: 0 30px 66px rgba(0,0,0,.55) !important;
          transform: translateY(-3px);
        }
        [data-testid="stMainBlockContainer"] .rl-chip { background: rgba(255,255,255,.2); color: #fff; }

        /* Glass buttons on the cards. */
        [data-testid="stMainBlockContainer"] .stButton > button {
          background: rgba(255,255,255,.12); color: #fff; border-color: rgba(255,255,255,.32);
          backdrop-filter: blur(8px); box-shadow: none;
        }
        [data-testid="stMainBlockContainer"] .stButton > button:hover {
          background: rgba(255,255,255,.22); color: #fff; border-color: rgba(255,255,255,.6); transform: translateY(-2px);
        }
        [data-testid="stMainBlockContainer"] .stButton > button p { color: #fff !important; }

        /* Compact, tidy file uploader: hide the sprawling drag zone, keep a clean button. */
        [data-testid="stMainBlockContainer"] [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
        [data-testid="stMainBlockContainer"] [data-testid="stFileUploaderDropzone"] {
          background: rgba(255,255,255,.07) !important;
          border: 1px solid rgba(255,255,255,.26) !important;
          border-radius: 12px !important;
          padding: .5rem !important;
          min-height: unset !important;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stFileUploaderDropzone"] button {
          width: 100%;
          background: rgba(255,255,255,.16) !important; color: #fff !important;
          border: 1px solid rgba(255,255,255,.45) !important;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stFileUploaderDropzone"] button:hover {
          background: rgba(255,255,255,.28) !important;
        }

        /* Slim résumé bar with a clear accent upload button. */
        .rl-resume-line { display: flex; align-items: center; gap: .6rem; flex-wrap: nowrap; min-width: 0; }
        .rl-resume-now {
          color: rgba(255,255,255,.5) !important; font-size: .72rem; font-weight: 700;
          text-transform: uppercase; letter-spacing: .06em; white-space: nowrap;
        }
        .rl-resume-name {
          color: #fff !important; font-weight: 700; font-size: 1rem;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] { padding: .65rem 1.1rem !important; }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"] {
          background: transparent !important; border: none !important; padding: 0 !important; min-height: unset !important;
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"] button {
          width: 100% !important; min-height: 2.9rem; font-size: 0 !important;
          background: linear-gradient(120deg, var(--rl-accent), var(--rl-accent-2)) !important;
          color: #fff !important; border: none !important;
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"] button::after {
          font: 700 .95rem "Inter", sans-serif; white-space: normal; line-height: 1.2;
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"] button:enabled::after {
          content: "✨ Get personalized results — upload your résumé!";
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"] button:disabled::after {
          content: "🔒 Upload your résumé";
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"] button:enabled:hover {
          filter: brightness(1.08); transform: translateY(-1px);
        }
        /* Demo only (the uploader is disabled there): explain why on hover. */
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"]:has(button:disabled) {
          position: relative;
        }
        [data-testid="stMainBlockContainer"] [class*="st-key-rlcard_upload"] [data-testid="stFileUploaderDropzone"]:has(button:disabled):hover::before {
          content: "This is the public demo — it explores a sample résumé. To upload your own and get personalized results, run ResumeLens yourself (clone the repo on GitHub).";
          position: absolute; bottom: calc(100% + 8px); left: 0; right: 0;
          background: rgba(18,16,14,.97); color: #fff;
          padding: .65rem .85rem; border-radius: 10px;
          font: 500 .8rem "Inter", sans-serif; line-height: 1.45;
          box-shadow: 0 12px 34px rgba(0,0,0,.45); z-index: 60;
        }
        /* Make the "?" help icon clearly visible on the dark card. */
        [data-testid="stMainBlockContainer"] [data-testid="stTooltipIcon"] svg { fill: rgba(255,255,255,.8) !important; }
        </style>
        <div class="rl-home-bg"></div>
        """,
        unsafe_allow_html=True,
    )


def set_sidebar(collapsed: bool) -> None:
    """Drive Streamlit's NATIVE sidebar collapse/expand, so the open/close
    controls and slide animation stay intuitive: collapsed on the landing,
    expanded inside a tool. No-ops when already in the desired state."""
    want = "collapsed" if collapsed else "expanded"
    _html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          const want = "{want}";
          function apply() {{
            const opener = doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
            const isCollapsed = !!(opener && opener.offsetParent !== null);
            if (want === "collapsed" && !isCollapsed) {{
              const close = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                         || doc.querySelector('[data-testid="stSidebarCollapseButton"]');
              if (close) close.click();
            }} else if (want === "expanded" && isCollapsed) {{
              const open = opener.querySelector('button') || opener;
              if (open) open.click();
            }}
          }}
          setTimeout(apply, 60);
        }})();
        </script>
        """,
        height=0,
    )


def enter_page(page_id: str, sidebar_collapsed: bool) -> None:
    """Run once on a real page change: set the native sidebar state (collapsed on
    the landing, expanded in a tool) and replay the page-in motion. Because it only
    fires when the page id changes, in-page reruns don't fight the user — they can
    still toggle the sidebar themselves and it stays put until they navigate."""
    if st.session_state.get("_rl_page") == page_id:
        return
    st.session_state["_rl_page"] = page_id
    # On the landing the sidebar is hidden by CSS, so we never toggle it there
    # (expand=false); on tool pages we expand it if it's currently collapsed.
    expand = "false" if sidebar_collapsed else "true"
    _html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          const expand = {expand};
          setTimeout(() => {{
            if (expand) {{
              const sb = doc.querySelector('[data-testid="stSidebar"]');
              const collapsed = !sb || sb.offsetWidth < 40;
              if (collapsed) {{
                const t = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                       || doc.querySelector('[data-testid="stSidebarCollapseButton"]');
                if (t) t.click();
              }}
            }}
            const el = doc.querySelector('[data-testid="stMainBlockContainer"]');
            if (el) {{ el.style.animation='none'; void el.offsetWidth;
                      el.style.animation='rl-page-in .6s cubic-bezier(.16,1,.3,1) both'; }}
          }}, 50);
        }})();
        </script>
        """,
        height=0,
    )


def render_home_hero(resume_name: str) -> None:
    st.markdown(
        f"""
        <header class="rl-hero-overlay">
          <div class="rl-brandmark"><span>🔍</span> ResumeLens</div>
          <div class="rl-kicker">Resume intelligence, with receipts</div>
          <h1>See beyond the résumé.</h1>
          <p>Retrieve the strongest evidence, watch how it was found, and keep every
             answer tied to the source. Currently exploring: <b>{resume_name}</b>.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_editorial_image(filename: str, eyebrow: str, title: str, caption: str) -> None:
    st.markdown(
        f"""
        <section class="rl-editorial-image" style="--editorial-image: url('/app/static/{filename}')">
          <div class="rl-editorial-image-copy">
            <div class="rl-kicker">{eyebrow}</div>
            <h3>{title}</h3>
            <p>{caption}</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="rl-section-intro">
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_usage(usage) -> None:
    """Show the token usage + a rough cost estimate for one Claude response."""
    if usage is None:
        return
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cread = getattr(usage, "cache_read_input_tokens", 0) or 0
    cwrite = getattr(usage, "cache_creation_input_tokens", 0) or 0

    p = config.PRICE_PER_1M
    cost = (
        inp * p["input"]
        + out * p["output"]
        + cread * p["cache_read"]
        + cwrite * p["cache_write"]
    ) / 1_000_000

    st.caption(
        f"🧾 **tokens** — input `{inp:,}` · output `{out:,}` "
        f"· cache read `{cread:,}` · cache write `{cwrite:,}`  ·  ≈ **${cost:.4f}**  \n"
        f"output is where reasoning effort shows up; cache read = the resume served "
        f"cheaply (only when the cached prefix exceeds the model's ~4k-token minimum)."
    )
