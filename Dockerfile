# ResumeLens on Hugging Face Spaces (Docker SDK).
# HF removed Streamlit as a native SDK, so we run it ourselves in a container.
FROM python:3.11-slim

# Hugging Face Spaces runs the container as a non-root user (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Public demo: lock to bundled sample résumés (no upload/paste).
    DEMO_MODE=1 \
    # Cache downloaded models under the writable HOME.
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers

WORKDIR /home/user/app

# Install Python deps first for better layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy the app source.
COPY --chown=user . .

# Hugging Face Spaces routes external traffic to this port.
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
