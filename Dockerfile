FROM python:3.12-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends 'ffmpeg=7:7.1.3-0+deb13u1' && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY LICENSE README.md pyproject.toml .
COPY bootlegger/ bootlegger/

RUN uv pip install --system --no-cache .

EXPOSE 8000

CMD ["bootlegger"]
