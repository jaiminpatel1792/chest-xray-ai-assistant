FROM python:3.10-slim

# Install system libraries needed by OpenCV
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user src ./src
COPY --chown=user models ./models
COPY --chown=user configs ./configs
COPY --chown=user start.sh ./start.sh
COPY --chown=user README.md ./README.md
COPY --chown=user .streamlit ./.streamlit

RUN chmod +x ./start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["./start.sh"]