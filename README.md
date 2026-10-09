# b60-dual-fixes

Two workarounds we run on a pair of Intel Arc Pro B60 cards serving vLLM with tensor parallel 2. Neither is in Intel's `llm-scaler-vllm` image.

1. `shim/` is a small `LD_AUDIT` library that stops Level Zero from mirroring every allocation into host RAM for the other card.
2. `patches/` is a patch to vLLM's `xpu_communicator.py` that pads all-reduce and all-gather messages to a fixed set of sizes.

Everything here was tested on one software stack, listed below. It may behave differently on others.

## Tested on

| | |
|---|---|
| GPUs | 2x Intel Arc Pro B60 24 GB, PCIe Gen3 x8 each, peer-to-peer through the CPU |
| Host | Ryzen 5 5600, X470 board, 64 GB RAM (an earlier Z690 box was used for the first runs) |
| OS | Ubuntu 24.04.4, kernel 6.17.0-1009-intel, Intel offline installer 26.18.8.2, GuC 70.60.0 |
| Image | `intel/llm-scaler-vllm:0.26.0-b2` |
| Inside it | vLLM `0.26.1.dev0+g568afb3a1.d20260907.xpu`, torch `2.12.0+xpu`, vllm-xpu-kernels `0.1.11.2.dev0+ga692986.d20260907` |
| Model | Qwen3.8-27B, GPTQ int4, TP=2 |

The patch was made against the copy of `xpu_communicator.py` inside that image (md5 `777f24059a6493d8f20c27cb1cc13a7d`). It applies cleanly to that file and compiles. We have not tried it on any other vLLM build.

## The residency shim

**Problem.** With peer-to-peer on, the Level Zero runtime calls `zeContextMakeMemoryResident` for every device allocation on the other card as well. The xe driver backs each of those with a host-memory allocation of the same size. This is intel/compute-runtime#968. In our case it showed up as about 17 GB of pinned host RAM per card at engine start on the first machine (32 GB of RAM, host OOM), and 24 to 26 GiB per card on the X470 with the stock image.

**What the shim does.** It records which device owns each `zeMemAllocDevice` result and returns success without doing anything when `zeContextMakeMemoryResident` is called for a different device. Peer transfers in oneCCL go through IPC handles, which this does not touch.

**What we measured.**

- A test that makes 4 GB of device allocations cost 4,383 MiB of host RAM without the shim and 288 MiB with it.
- On the X470, pinned host RAM went from 24 to 26 GiB per card to 11 GiB per card. That figure is for the shim together with the oneCCL `*_SIMPLE_THRESHOLD` settings below; we did not separate the two.
- It made no difference to the KV cache size we got (657k tokens with and without it).
- The full configuration with it on had 0 hangs in 13 engine starts, and took part in an 8.7 hour agent-load run (alternating with an eager-mode config) with no stalls.

**Why `LD_AUDIT`.** `LD_PRELOAD` does nothing here, because UMF and the Level Zero v2 adapter `dlopen` the loader and look each function up with `dlsym`. Interposing `dlsym` itself broke the Intel graphics compiler (it reported the program was built for 2 devices and died with an internal compiler error). Hooking at symbol-binding time avoids both.

The v2 adapter also reaches most entry points through the `zeGet*ProcAddrTable` function tables, which symbol binding never sees. The shim patches the context and memory tables when the v2 adapter asks for them. Those table slots are addressed by index (see the comments in the source), so a Level Zero loader that reorders the tables would break it.

**Build and use.**

```sh
./shim/build.sh        # needs gcc; produces shim/ze_residency_audit.so
docker run ... \
  -v $PWD/shim/ze_residency_audit.so:/opt/ze_residency_audit.so:ro \
  -e LD_AUDIT=/opt/ze_residency_audit.so \
  -e ZE_RESIDENCY_SHIM_LOG=1 \
  ...
```

`ZE_RESIDENCY_SHIM_LOG=1` prints counters of skipped and passed calls to stderr. Leave it out for quiet runs.

The oneCCL settings we ran alongside it:

```
CCL_TOPO_P2P_ACCESS=1
CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1073741824
CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=1073741824
CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=1073741824
```

**Limits.** Only the allocations made after the library loads are tracked. If a kernel really does touch a peer allocation that was never made resident, we would expect a fault. We did not see one, but we only ran our own workloads. The source is under 200 lines, so read it before trusting it.

## The all-reduce padding patch

**Problem.** This only applies when all-reduce runs on oneCCL's host-staged path (`CCL_TOPO_P2P_ACCESS=0`). That path keeps one host staging buffer per exact message size in `ccl::regular_buffer_cache` and never evicts. Message size follows the token count of each prefill chunk, so varied prompts keep adding buffers, up to about 84 MB each. On 12 varied prompts we saw about 1.15 GB of growth per worker.

Turning the cache off with `CCL_BUFFER_CACHE=0` stops the growth but halved decode speed (43 to 20 tok/s in our run), because every sync then allocates and registers a new buffer.

**What the patch does.** When `VLLM_XPU_AR_PAD=1`, eager all-reduce messages above a threshold are zero-padded to the next power of two (up to 2^21 elements) or the next multiple of 2^21, reduced, and the original length is copied back. The in-graph `all_gather` pads its row count the same way. Padding with zeros does not change the reduced values, and greedy output for a 9,125-token prompt was token-identical with padding on and off. Small messages (up to 64 x 5120 elements) are not padded, since padding them cost about 20% of decode speed.

Extra knobs: `VLLM_XPU_AR_PAD_MIN`, `VLLM_XPU_AR_PAD_GRAN`, and `VLLM_XPU_COLL_TRACE=1`, which logs a stack for any collective over 4 MB that is not a bucket size.

**What it did and did not do.** The padded runs on the host-staged path still grew: 863 to 1,394 MB per worker on the first pass and 232 to 288 MB on the second, against 1.1 GB for the unpadded run. So padding alone did not flatten the host path in those tests. What did stop the growth was moving all-reduce to the peer-to-peer path, where oneCCL made zero buffer-cache lookups (against 48,000 on the host path) and a 41-request sweep grew workers by about 10 MB. We left padding on for that setup and did not isolate its contribution there. If you run peer-to-peer, you probably do not need this patch. We include it because the measurements above came from it.

**Apply.**

```sh
cd /opt/venv/lib/python3.12/site-packages     # inside the image
patch -p1 < xpu-communicator-bucketed-allreduce.patch
```

Or leave the image alone and mount a patched copy of the file over the original, read-only, which is what we do. Then set `-e VLLM_XPU_AR_PAD=1`. Without that variable the patched file behaves like the original.

## Licence

Apache-2.0, see `LICENSE` and `NOTICE`. The patch is a derivative of vLLM's `xpu_communicator.py` and keeps vLLM's Apache-2.0 licence. intel/llm-scaler is also Apache-2.0; nothing from it is included.

Made by [LocalXPU](https://github.com/LocalXPU).
