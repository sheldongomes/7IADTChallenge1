# ---- Base image -------------------------------------------------
FROM python:3.12-slim

# ---- Instalar ferramentas essenciais ---------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Atualizar pip, setuptools, wheel --------------------------
RUN pip install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel

# ---- Criar usuário não-root ------------------------------------
ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# ---- Mudar para o usuário ANTES de criar pastas ---------------
USER $USERNAME
WORKDIR /app

# ---- Copiar e instalar dependências ----------------------------
COPY --chown=$USERNAME:$USERNAME requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Criar pastas COMO appuser (agora tem permissão) ----------
RUN mkdir -p models results

# ---- Copiar código com dono correto ---------------------------
COPY --chown=$USERNAME:$USERNAME . .

# ---- Tornar main.py executável ---------------------------------
RUN chmod +x main.py

# ---- Rodar init + API ------------------------------------------
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python main.py && python /app/analysis/eda.py && python /app/analysis/explainability.py && python /app/analysis/modeling.py && python /app/utils/html_results.py && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"]