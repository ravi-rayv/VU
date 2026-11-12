"""Entry point for Streamlit Community Cloud and the dev container, which look
for ``streamlit_app.py`` in the repository root.

The portal itself lives in ``app/streamlit_app.py``; this file only runs it, so
there is a single copy of the app to maintain.

Run locally:   streamlit run streamlit_app.py   (or app/streamlit_app.py)
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py"),
               run_name="__main__")
