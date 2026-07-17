FROM python:3.11-slim

WORKDIR /app

# System libraries docling needs (opencv and image processing).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Runtime deps resolve from pyproject.toml rather than a list hand-maintained
# here. The previous list named only fastapi/uvicorn/pydantic-settings/docling
# and omitted pypdf and openpyxl -- both imported at module load -- so the image
# died on `from parser_service.main import app`. Nothing built the image in CI,
# so that stayed invisible. One source of truth for deps, plus an image job that
# actually imports the app, is what stops it recurring.
#
# --extra-index-url pulls CPU-only PyTorch (torch arrives transitively via
# docling-ibm-models): this service does layout analysis on CPU, and the default
# wheels would drag in the entire NVIDIA CUDA stack for a GPU that is not there.
#
# Deliberately pip, not uv. That index also publishes an ancient certifi
# (2022.12.7); pip considers every index and takes the newest, while uv defaults
# to first-index and would pin certifi from the torch index, colliding with
# docling's certifi>=2024.7.4.
#
# The consequence is that this image cannot read uv.lock, so CI (uv) and this
# image (pip) resolve separately and DO diverge today: CI tests transformers
# 5.8.1 while this ships 5.14.1. Note which side is wrong -- this one is right.
# docling caps transformers <5.9.0 on darwin but <6.0.0 elsewhere, and uv's
# universal lock must satisfy every platform at once, so the macOS ceiling
# leaks onto Linux. pip resolves per-platform and correctly takes 5.14.1.
# Therefore do NOT "fix" this by copying uv.lock and running `uv sync --frozen`:
# against today's lock that ships the darwin-capped 5.8.1 to Linux, six minors
# backwards. Fix the lock first, then point the image at it.
#
# Scoping the index to torch alone via [[tool.uv.index]] explicit = true is the
# real answer, but it does NOT work as a drop-in: uv applies [tool.uv.sources]
# to DIRECT dependencies only, and torch is transitive (docling ->
# docling-ibm-models -> torch). Configure it with torch left transitive and uv
# exits 0, emits no warning, and locks CUDA torch anyway. torch and torchvision
# must be promoted to declared dependencies -- deps this service never imports --
# for the pin to bind at all. That is a deliberate change to what CI tests
# (transformers +6 minors on the code producing the bboxes we cite), so it wants
# its own PR and a re-baselined suite, not a side effect of a repo split.
COPY pyproject.toml README.md ./
COPY parser_service/ ./parser_service/
RUN pip install --no-cache-dir . \
    --extra-index-url https://download.pytorch.org/whl/cpu

ENV OMP_NUM_THREADS=4
ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

EXPOSE 8001

CMD ["uvicorn", "parser_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
