FROM base AS base

FROM docker.io/library/buildpack-deps:bookworm AS builder

WORKDIR /app

COPY . .
COPY --from=base /pkgs ./pkgs

RUN apt-get update -y && apt-get install -y --no-install-recommends jq cmake libsasl2-dev && \
  rm -rf /var/lib/apt/lists/* && \
  make release

FROM docker.io/library/debian:bookworm-slim

ENV TZ=Asia/Shanghai

RUN apt-get update -y && apt-get install -y --no-install-recommends ca-certificates && \
  rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/bin/aipi /usr/bin

CMD ["spider", "-c", "/etc/spider-cplusplus/config.yaml"]
