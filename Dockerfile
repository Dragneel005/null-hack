# Push-up Form Coach - Hugging Face Docker Space (free CPU tier).
# bookworm pinned on purpose: the shared-lib package names below are bookworm names.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg for video re-encoding; GL/GLib/EGL libs for OpenCV + MediaPipe on slim images;
# portaudio because mediapipe depends on sounddevice; DejaVu font for the overlay text.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 libportaudio2 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000. Create the app + writable dirs as root and hand
# them to the user, so this works whether or not the builder chowns WORKDIR.
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app/outputs/uploads /home/user/app/data \
    && chown -R user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    LEADERBOARD_DB=/home/user/app/data/leaderboard.db \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# XSRF/CORS off: otherwise st.file_uploader fails with "AxiosError 403" behind the Spaces proxy.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]
