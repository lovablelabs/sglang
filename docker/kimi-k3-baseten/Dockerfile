ARG BASE_PLATFORM=linux/amd64
ARG SERVING_IMAGE=docker.io/lmsysorg/sglang@sha256:a8eedda74b54f7dfecc445ee66d298f8d8f6f50192201a53b0a6dd7571ff8d26
FROM --platform=${BASE_PLATFORM} docker.io/library/rust@sha256:f845b308e3104231c98dba2dab952ebd54e35cd6ab8ef4a230ecb38e2315f20c AS rust
FROM --platform=${BASE_PLATFORM} ${SERVING_IMAGE} AS tokenizer-build
USER root
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:${PATH}
RUN /opt/sglang/bin/python3 -m pip install --no-cache-dir maturin==1.16.0
WORKDIR /build/basetenkenizer
RUN git init . && git remote add origin https://github.com/basetenlabs/basetenkenizer.git && git fetch --depth 1 origin 792da9f0a74350318ffa03a6b455c3d582c9e861 && git checkout --detach FETCH_HEAD
COPY docker/kimi-k3-baseten/python-whitespace-parity.patch /tmp/whitespace.patch
RUN git apply --check /tmp/whitespace.patch && git apply /tmp/whitespace.patch && /opt/sglang/bin/maturin build --release --locked --interpreter /opt/sglang/bin/python3 --out /wheels

FROM --platform=${BASE_PLATFORM} ${SERVING_IMAGE}
USER root
COPY --from=tokenizer-build /wheels /tmp/baseten-wheels
RUN /opt/sglang/bin/python3 -m pip install --force-reinstall --no-deps /tmp/baseten-wheels/*.whl && rm -rf /tmp/baseten-wheels
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/
ENV SGLANG_CACHE_DIR=/tmp/sglang-cache RAYON_NUM_THREADS=1 SGLANG_ENABLE_PP_SPEC=0
LABEL lovable.sglang.base-commit=c1c287a42928495166e14dbcef31b9e1f1dac199
LABEL lovable.baseten.base-commit=792da9f0a74350318ffa03a6b455c3d582c9e861
LABEL lovable.baseten.fix=python-whitespace-parity
