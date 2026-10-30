# Kimi K3 with Baseten encoding

This build adds `--tokenizer-backend baseten` to the Kimi serving revision `c1c287a42928495166e14dbcef31b9e1f1dac199`. It preserves the existing BF16 key-packing fix and the eight-B300 K3/DSpark launch configuration. Build and run on Linux AMD64 with the NVIDIA container runtime.

From the repository root:

```sh
docker build --platform linux/amd64 -f docker/kimi-k3-baseten/Dockerfile -t kimi-k3-baseten:local .
```

The checkpoint directory must contain `Kimi-K3-NVFP4/` and `Kimi-K3-DSpark/`, including their weights, tokenizer files and custom Python files. The saved revisions are `nvidia/Kimi-K3-NVFP4@b2428a0b83a8b712ff2e1a8448a103d4175341f1` and `RadixArk/Kimi-K3-DSpark@3c5bac301d9cf392706189d82ed947feca6c2f0f`. This recipe does not download model weights.

```sh
python3 docker/kimi-k3-baseten/run_server.py --image kimi-k3-baseten:local --checkpoints /srv/checkpoints
```

Add `--print-command` to inspect the exact Docker command. The launcher preserves the saved server arguments in `server-argv.json`, mounts checkpoints read-only, and serves on `127.0.0.1:30000` on the GPU host. The existing reasoning parser, tool parser, tensor/expert parallelism, DSpark settings and 262,144-token context are retained.

The backend reuses the original Kimi ranks, regex and special-token map. It keeps Kimi's renderer and decoder, preserves segment boundaries and special-token policies, and uses the original encoder for malformed Unicode strings. Unsupported tokenizer families fail explicitly. Multimodal preprocessing workers use `spawn`; native encoders are initialized lazily and reconstructed after copying or pickling.

The Dockerfile pins Baseten source and applies the Python-whitespace compatibility fix for U+001C–U+001F. It uses one Rayon thread, matching the qualified CPU path. The decoder and direct-input-ID bypass retain their existing behavior.

CPU qualification on the pinned serving image passed all 808 exact-ID cases, ten malformed-Unicode policy/boundary checks, chat/tool formatting, batch padding, truncation, deepcopy, pickle and spawned-worker checks. The actual SGLang tokenizer loader and K3 processor loader passed, and the launch CLI recognizes `baseten`. A Linux AMD64 wheel was built successfully. Full GPU server startup, real image processing and end-to-end concurrency latency have not been qualified by these CPU checks.
