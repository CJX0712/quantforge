# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Single-threaded BLAS is required for the bit-identical determinism guarantee:
# multi-threaded GEMM changes summation order between runs.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY quantforge ./quantforge
COPY examples ./examples
COPY tests ./tests

RUN pip install --no-cache-dir --no-deps -e .

# Fail fast if the package cannot be imported or the suite is broken.
RUN python -c "import quantforge; print(quantforge.__version__)" \
 && python -m pytest tests -q

ENTRYPOINT ["python", "-m", "quantforge"]
CMD ["demo", "--outdir", "results", "--progress"]