# syntax=docker/dockerfile:1

# ========================================
# Builder stage: build the VIRDA wheel
# ========================================
FROM python:3.14-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /uvx /bin/

WORKDIR /build

# Copy dependency files and source code
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Build sdist + wheel
RUN uv build

# ========================================
# Runtime stage
# ========================================
FROM python:3.14-slim

ARG USERNAME="appuser"
ARG USER_UID=1000
ARG USER_GID=1000
ARG APP_DIR="/workspace"
ARG LOGS_DIR="/var/log/app"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /uvx /bin/

# Setup user, app dir and logs dir
RUN mkdir -p "/home/${USERNAME}" && \
    groupadd -r -g ${USER_GID} ${USERNAME} && \
    useradd -r -d "/home/${USERNAME}" -u ${USER_UID} -g ${USER_GID} ${USERNAME} && \
    chown ${USERNAME}:${USERNAME} "/home/${USERNAME}" && \
    mkdir -p "${APP_DIR}" && \
    chmod 750 "${APP_DIR}" && \
    chown ${USERNAME}:${USERNAME} "${APP_DIR}" && \
    mkdir -p "${LOGS_DIR}" && \
    chown ${USERNAME}:${USERNAME} "${LOGS_DIR}"

WORKDIR "${APP_DIR}"
USER ${USERNAME}

# Point the venv to the project directory
ENV VIRTUAL_ENV="${APP_DIR}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Install dependencies from the lockfile (project itself comes from the wheel)
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

# Install the built VIRDA wheel
COPY --from=builder --chown=${USERNAME}:${USERNAME} /build/dist/virda-*.whl ./
RUN uv pip install --no-deps virda-*.whl && rm virda-*.whl

# Default command
ENTRYPOINT []
CMD ["virda"]
