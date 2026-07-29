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

# Copy dependency files
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock* ./

# Cache only dependencies
RUN uv sync --frozen --no-dev

# Copy source code and tests
COPY --chown=${USERNAME}:${USERNAME} src/ ./src/
COPY --chown=${USERNAME}:${USERNAME} tests/ ./tests/

# Set Python path
ENV PATH="${APP_DIR}/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT []
CMD ["python", "-m", "src.virda.main"]
