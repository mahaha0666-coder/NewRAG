"""Launch one local model in the existing GPU environment, without remote code."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["paddle", "visual", "mech", "visual4b"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-image-pixels", type=int, default=1048576)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-devices",
                        help="Visible physical GPUs; visual supports two GPUs or one with CPU offload")
    parser.add_argument("--cpu-offload-gib", type=float, default=0,
                        help="Visual model weight offload to host memory (GiB); preserves model weights")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        help="GPU memory budget fraction; lower when sharing a GPU with another workload")
    parser.add_argument("--max-num-seqs", type=int,
                        help="Maximum concurrent sequences (visual4b=4, paddle=2, otherwise 1)")
    parser.add_argument("--kv-cache-gib", type=float, help="Fixed KV cache budget for drawing inference")
    parser.add_argument("--model-path", type=Path,
                        help="Override the local model directory (useful for the small-model route)")
    parser.add_argument("--served-model-name", help="Override the OpenAI model id")
    parser.add_argument("--decode-graph", action="store_true",
                        help="Capture single-request GPU decoding for the 27B visual model")
    args = parser.parse_args()
    paddle = args.model == "paddle"
    mech = args.model == "mech"
    small = args.model == "visual4b"
    visual = args.model == "visual"
    devices = (args.gpu_devices or ("0,1" if visual else "0")).split(",")
    if not devices or any(not device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        parser.error("--gpu-devices must contain distinct numeric GPU indices")
    if not 0 <= args.cpu_offload_gib <= 64:
        parser.error("--cpu-offload-gib must be between 0 and 64")
    if args.cpu_offload_gib and not visual:
        parser.error("--cpu-offload-gib is supported only for the visual model")
    if visual and (len(devices) not in (1, 2) or (len(devices) == 1 and not args.cpu_offload_gib)):
        parser.error("27B visual inference requires two GPUs or explicit CPU offload for one GPU")
    if paddle and len(devices) != 1:
        parser.error("Paddle inference requires exactly one visible GPU")
    kv_cache_gib = args.kv_cache_gib if args.kv_cache_gib is not None else (8 if paddle else 2 if small else None)
    if kv_cache_gib is not None and not .5 <= kv_cache_gib <= 32:
        parser.error("--kv-cache-gib must be between 0.5 and 32")
    if args.max_image_pixels < 65536:
        parser.error("--max-image-pixels must be at least 65536")
    default_model_len = 32768 if paddle or small else 16384 if mech else 49152
    max_model_len = args.max_model_len or default_model_len
    if not 8192 <= max_model_len <= 262144:
        parser.error("--max-model-len must be between 8192 and 262144")
    max_num_seqs = args.max_num_seqs or (2 if paddle else 4 if small else 1)
    if not 1 <= max_num_seqs <= 8:
        parser.error("--max-num-seqs must be between 1 and 8")
    if args.decode_graph and (not visual or max_num_seqs != 1 or args.cpu_offload_gib):
        parser.error("--decode-graph requires GPU-resident visual inference with --max-num-seqs 1")
    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 0.85 if visual else 0.18 if paddle or mech else 0.52
    if not 0 < gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1")
    port = args.port or (18022 if paddle else 18023 if small else 18021)
    python = Path("/home/userroot/miniconda3/envs/vllm-qwen38/bin/python")
    env = dict(os.environ)
    env.update(PATH=str(python.parent) + ":/usr/local/cuda/bin:" + env.get("PATH", ""),
               VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="2",
               CUDA_VISIBLE_DEVICES=",".join(devices))
    if paddle:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    default_model = Path("/home/userroot/models/" + (
        "PaddleOCR-VL-1.6" if paddle else "MechVL-4B-RL" if mech
        else "Qwen3.5-4B" if small else "Qwen3.8-27B-INT4"))
    default_name = "PaddleOCR-VL-1.6" if paddle else "drawing-mech" if mech else "drawing-visual-4b" if small else "drawing-visual"
    model_path = args.model_path or default_model
    served_name = args.served_model_name or default_name
    if args.model_path is not None and not model_path.is_dir():
        parser.error(f"model directory does not exist: {model_path}")
    command = [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model",
               str(model_path), "--served-model-name", served_name,
               "--host", "127.0.0.1", "--port", str(port), "--max-model-len",
               str(max_model_len), "--max-num-seqs", str(max_num_seqs),
               "--gpu-memory-utilization", str(gpu_memory_utilization),
               "--limit-mm-per-prompt", '{"image":1}' if paddle else '{"image":5}',
               "--mm-encoder-attn-backend", "TORCH_SDPA"]
    if args.decode_graph:
        # Only capture the recurrent one-token decode. Prefill still accepts
        # all supported prompt lengths, image sizes and input combinations.
        command += ["--compilation-config", json.dumps({
            "mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        }), "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"]
    else:
        command.append("--enforce-eager")
    if visual:
        command += ["--tensor-parallel-size", str(len(devices)), "--no-enable-flashinfer-autotune"]
    if args.cpu_offload_gib:
        command += ["--cpu-offload-gb", str(args.cpu_offload_gib)]
    if kv_cache_gib is not None:
        command += ["--kv-cache-memory-bytes", str(int(kv_cache_gib * 1024 ** 3))]
    if not paddle:
        command += ["--mm-processor-kwargs", json.dumps({"size": {"longest_edge": args.max_image_pixels, "shortest_edge": 65536}})]
    if mech:
        command += ["--chat-template", "/home/userroot/models/MechVL-4B-RL/chat_template.jinja", "--dtype", "bfloat16"]
    os.execve(python, command, env)


if __name__ == "__main__":
    main()
