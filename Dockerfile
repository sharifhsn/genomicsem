# syntax=docker/dockerfile:1
#
# Multi-stage build for genomicsem.
#
# Stages
# ------
#   base     — Rust toolchain + build tools + sccache + cargo-chef
#   planner  — generate cargo-chef recipe (dependency fingerprint)
#   builder  — compile deps then source; both stages use BuildKit cache mounts
#   runtime  — minimal Debian image with only the stripped binary
#
# BuildKit cache mounts keep the cargo registry and sccache artifacts on the
# build host between runs, so individual crate recompilations are skipped when
# only application source changes.

# ── base: toolchain + tools ───────────────────────────────────────────────────
FROM rust:1 AS base

# Use system OpenBLAS (default feature) and link dynamically.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libbz2-dev \
        cmake \
        gfortran \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Compile sccache and cargo-chef once; cached as a Docker layer.
RUN cargo install sccache --locked && cargo install cargo-chef --locked

ENV RUSTC_WRAPPER=sccache \
    SCCACHE_DIR=/sccache \
    CARGO_INCREMENTAL=0

# ── planner: generate dependency recipe ──────────────────────────────────────
FROM base AS planner
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json

# ── builder: compile deps then source ────────────────────────────────────────
FROM base AS builder
WORKDIR /app

# Cook dependencies — cache-hit whenever recipe.json is unchanged.
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json

# Compile the application.
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release

# ── runtime: minimal final image ─────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# libgfortran5: Fortran runtime used by OpenBLAS.
# libopenblas0: runtime OpenBLAS shared library.
# libbz2-1.0: runtime bzip2 library.
# ca-certificates: needed for HTTPS downloads (LD score files, summary stats).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgfortran5 \
        libopenblas0 \
        libbz2-1.0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/genomicsem /usr/local/bin/genomicsem

ENTRYPOINT ["genomicsem"]
