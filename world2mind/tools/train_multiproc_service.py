"""
Multi-process / multi-port launcher for the World2Mind service (training use).

Spawns N independent single-GPU subprocesses of `start_service.py`, each with:
  - CUDA_VISIBLE_DEVICES set to a single physical GPU
  - --gpu_ids 0  (only GPU visible to that process)
  - --port base_port + gpu_id * instances_per_gpu + inst_idx
  - --workers_per_gpu 1
  - --max_cpu_concurrent 1

instances_per_gpu (default 1):
  - N=1: one process per GPU — same as original behavior.
  - N>1: N processes per GPU, each on its own port, sharing the physical GPU
    via CUDA driver time-slicing (Default compute mode, no MPS required).
    Benefit: CPU phases (mapping/AST/route ~5s) run truly in parallel,
    queue depth per port halved → lower tail latency under 32-concurrent rollout.
    GPU phase throughput unchanged (DA3+SAM3 saturates memory bandwidth).

Port assignment (backward-compatible):
  N=1: base_port + gpu_id  (identical to original)
  N>1: base_port + gpu_id * N + inst_idx

Why: removes all cross-process / cross-GPU GIL and threading.Lock contention.
True parallelism: 6 physical GPUs × N instances → 6*N independent processes,
each handling one request at a time with no shared state.

Startup: serial (one instance at a time, wait for /health with models loaded).
Shutdown: SIGINT/SIGTERM propagated to all children.
Resilience: a single child crashing (e.g. DA3/SAM3 segfault, OOM) is
auto-restarted in place; surviving children keep serving traffic. Each
instance has its own independent restart counter keyed by port number, so a
crash on one instance does not affect the restart accounting of sibling
instances on the same GPU.

Hung-worker watchdog: a native thread crash (libgomp "Thread creation failed"
→ SIGSEGV inside pycolmap/PyTorch) can leave a per-GPU threading.Lock
permanently held while the uvicorn process stays alive (health → 200 OK).
The watchdog detects requests in-flight longer than --hung_timeout_s (default
600s) and force-kills + restarts the child before the client's 1800s timeout
fires.  Disable with --hung_timeout_s 0.

Log routing:
  - Each child's stdout+stderr → logs/w2m_gpu{X}.log        (N=1)
                                  logs/w2m_gpu{X}_inst{Y}.log (N>1)
  - Main stdout shows only concise events:
      [gpu X]   READY at startup           (N=1)
      [gpu X.Y] READY at startup           (N>1)
      [gpu X]   START scene=Z              (when request arrives)
      [gpu X]   DONE  scene=Z duration=Ws  (when request completes)
      [gpu X]   ERR   ...                  (on failure)

Usage:
    python tools/train_multiproc_service.py \\
        --gpu_ids 0,1,2,3,4,5 \\
        --base_port 9100 \\
        --instances_per_gpu 1 \\
        --config config/default_config.yaml \\
        --log_dir demo/sft_scripts/grpo/logs \\
        --hung_timeout_s 600

Environment for clients:
    WORLD2MIND_MULTIPORT=1
    WORLD2MIND_BASE_PORT=9100
    WORLD2MIND_GPU_IDS=0,1,2,3,4,5
    WORLD2MIND_INSTANCES_PER_GPU=1   # match --instances_per_gpu value above
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START_SERVICE = PROJECT_ROOT / "start_service.py"


# ---------------------------------------------------------------------------
# Log tailer: parse child logs and emit concise events to main stdout
# ---------------------------------------------------------------------------

# model_service.py: `logger.info(f"[GPU {worker.gpu_id}] Cognitive map request: {request.scene_id}")`
_REQ_START = re.compile(r"Cognitive map request:\s*(\S+)")
# uvicorn access log on request end
_REQ_END = re.compile(r'"POST /cognitive_map HTTP/1\.1" (\d+)')
# error markers
_ERR = re.compile(r"(Cognitive map failed|ERROR - |Traceback \(most recent call last\))")


class ChildLogTailer(threading.Thread):
    """Reads a child process log file in real time and emits concise events."""

    def __init__(self, label: str, log_path: Path, emit_lock: threading.Lock):
        super().__init__(daemon=True)
        self._label = label          # e.g. "gpu 2" or "gpu 2.1"
        self._log_path = log_path
        self._emit_lock = emit_lock
        self._stop = threading.Event()
        # FIFO of (scene_id, start_time) for in-flight requests. Uvicorn handles
        # endpoints on a threadpool so multiple requests can log START before
        # the first one's END access-log line is emitted. We assume FIFO
        # completion order (single GPU lock + max_cpu_concurrent=1 serialize).
        from collections import deque
        self._inflight: "deque[tuple[str, float]]" = deque()
        self._inflight_lock = threading.Lock()

    def oldest_inflight_age(self) -> Optional[float]:
        """Return how long (seconds) the oldest in-flight request has been waiting.

        Returns None if no request is currently in flight.
        This is used by the hung-worker watchdog: if a request stays in-flight
        longer than `--hung_timeout_s`, the worker has deadlocked (e.g. a native
        thread crash left a GPU lock permanently held) and must be restarted.
        """
        with self._inflight_lock:
            if not self._inflight:
                return None
            return time.time() - self._inflight[0][1]

    def stop(self):
        self._stop.set()

    def _emit(self, msg: str):
        with self._emit_lock:
            print(f"[{self._label}] {msg}", flush=True)

    def run(self):
        # Wait until log exists
        while not self._log_path.exists() and not self._stop.is_set():
            time.sleep(0.5)
        if self._stop.is_set():
            return
        # Open with errors='replace' so non-UTF-8 bytes (ANSI escape codes,
        # binary content from ffmpeg error messages, etc.) never crash the tailer.
        with self._log_path.open("r", encoding="utf-8", errors="replace") as f:
            # Start from end (we only care about live events)
            f.seek(0, os.SEEK_END)
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                self._handle_line(line.rstrip())

    def _handle_line(self, line: str):
        m = _REQ_START.search(line)
        if m:
            scene = m.group(1)
            with self._inflight_lock:
                self._inflight.append((scene, time.time()))
                n = len(self._inflight)
            self._emit(f"START scene={scene} inflight={n}")
            return
        m = _REQ_END.search(line)
        if m:
            status = m.group(1)
            with self._inflight_lock:
                if self._inflight:
                    scene, start = self._inflight.popleft()
                    dur_str = f"{time.time() - start:.1f}s"
                else:
                    scene, dur_str = "?", "?"
                n = len(self._inflight)
            tag = "DONE" if status.startswith("2") else f"FAIL({status})"
            self._emit(f"{tag}  scene={scene} duration={dur_str} "
                       f"inflight={n}")
            return
        if _ERR.search(line):
            self._emit(f"ERR   {line[:180]}")


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

class ChildProcess:
    def __init__(self, gpu_id: int, instance_id: int, port: int,
                 popen: subprocess.Popen, log_path: Path, tailer: ChildLogTailer):
        self.gpu_id = gpu_id
        self.instance_id = instance_id   # 0..instances_per_gpu-1
        self.port = port
        self.popen = popen
        self.log_path = log_path
        self.tailer = tailer


def _make_label(gpu_id: int, instance_id: int, instances_per_gpu: int) -> str:
    """Return human-readable label for log emit.

    N=1 → "gpu X"   (backward-compatible with existing log parsers)
    N>1 → "gpu X.Y"
    """
    if instances_per_gpu == 1:
        return f"gpu {gpu_id}"
    return f"gpu {gpu_id}.{instance_id}"


def _make_log_path(log_dir: Path, gpu_id: int, instance_id: int,
                   instances_per_gpu: int) -> Path:
    """Return log file path for a child process.

    N=1 → w2m_gpu{X}.log   (backward-compatible)
    N>1 → w2m_gpu{X}_inst{Y}.log
    """
    if instances_per_gpu == 1:
        return log_dir / f"w2m_gpu{gpu_id}.log"
    return log_dir / f"w2m_gpu{gpu_id}_inst{instance_id}.log"


def _wait_ready(port: int, timeout_s: float, label: str, emit_lock: threading.Lock) -> bool:
    """Poll /health until workers report both DA3 and SAM3 loaded."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = data.get("workers") or []
                if workers and workers[0].get("da3_loaded") and workers[0].get("sam3_loaded"):
                    return True
        except Exception as e:
            last_err = e
        time.sleep(2.0)
    with emit_lock:
        print(f"[{label}] TIMEOUT waiting for /health (last={last_err})", flush=True)
    return False


def launch_child(gpu_id: int, instance_id: int, port: int,
                 instances_per_gpu: int, config_path: str,
                 log_dir: Path, emit_lock: threading.Lock) -> ChildProcess:
    label = _make_label(gpu_id, instance_id, instances_per_gpu)
    log_path = _make_log_path(log_dir, gpu_id, instance_id, instances_per_gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate previous log
    log_path.write_text("")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Inside the child, the single visible GPU is index 0.
    # Multiple instances per GPU share the same physical device via Default
    # compute mode (CUDA driver time-slicing) — no EXCLUSIVE_PROCESS needed.
    cmd = [
        sys.executable,
        str(START_SERVICE),
        "--config", config_path,
        "--gpu_ids", "0",
        "--port", str(port),
        "--workers_per_gpu", "1",
        "--max_cpu_concurrent", "1",
    ]

    log_fp = log_path.open("ab", buffering=0)
    popen = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # separate process group → signal propagation
    )
    tailer = ChildLogTailer(label=label, log_path=log_path, emit_lock=emit_lock)
    tailer.start()
    with emit_lock:
        print(f"[{label}] launched pid={popen.pid} port={port} log={log_path}", flush=True)
    return ChildProcess(gpu_id=gpu_id, instance_id=instance_id, port=port,
                        popen=popen, log_path=log_path, tailer=tailer)


def shutdown(children: List[ChildProcess], emit_lock: threading.Lock):
    with emit_lock:
        print("[main] shutting down children...", flush=True)
    for c in children:
        try:
            if c.popen.poll() is None:
                c.popen.terminate()
        except Exception:
            pass
    # Grace period
    deadline = time.time() + 10.0
    for c in children:
        remaining = max(0.0, deadline - time.time())
        try:
            c.popen.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                c.popen.kill()
            except Exception:
                pass
    for c in children:
        c.tailer.stop()
    with emit_lock:
        print("[main] all children stopped", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, required=True,
                        help="Comma-separated physical GPU IDs, e.g. 0,1,2,3,4,5")
    parser.add_argument("--base_port", type=int, default=9100)
    parser.add_argument("--instances_per_gpu", type=int, default=1,
                        help="Number of independent service processes to launch per GPU. "
                             "Each gets its own port. "
                             "N=1 (default): one process per GPU, identical to original behavior. "
                             "N=2: two processes per GPU on consecutive ports; GPU phases are "
                             "time-sliced by the CUDA driver (Default compute mode, no MPS needed); "
                             "benefit is CPU-phase parallelism and halved queue depth per port.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--log_dir", type=str,
                        default=str(PROJECT_ROOT / "demo" / "sft_scripts" / "grpo" / "logs"))
    parser.add_argument("--ready_timeout_s", type=float, default=600.0,
                        help="Seconds to wait for /health during (re)start before giving up")
    parser.add_argument("--max_restart_fails", type=int, default=10,
                        help="Abort only if the SAME instance fails to come back up "
                             "this many times in a row. A successful restart "
                             "resets the counter. Healthy crashes (new process "
                             "starts up fine) never count toward this limit.")
    parser.add_argument("--restart_cooldown_s", type=float, default=5.0,
                        help="Wait this long before spawning a new process after "
                             "a crash (lets OS clean up fds, GPU reset, etc.).")
    parser.add_argument("--flap_window_s", type=float, default=120.0,
                        help="If a newly-restarted child crashes within this "
                             "many seconds of becoming READY, count it as a "
                             "partial failure (fail_streak += 0.5) to catch "
                             "crash-looping without blocking transient crashes.")
    parser.add_argument("--hung_timeout_s", type=float, default=600.0,
                        help="If a /cognitive_map request has been in-flight for "
                             "longer than this many seconds, the worker is considered "
                             "hung (e.g. a native thread crash left a GPU lock "
                             "permanently held) and is force-killed and restarted. "
                             "Set to 0 to disable the hung-worker watchdog.")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    instances_per_gpu = max(1, args.instances_per_gpu)
    log_dir = Path(args.log_dir)
    emit_lock = threading.Lock()

    total_instances = len(gpu_ids) * instances_per_gpu
    with emit_lock:
        print(f"[main] starting {total_instances} worker(s): "
              f"{len(gpu_ids)} GPU(s) × {instances_per_gpu} instance(s)/GPU", flush=True)

    children: List[ChildProcess] = []
    stopping = threading.Event()

    def _signal_handler(signum, frame):
        with emit_lock:
            print(f"[main] received signal {signum}", flush=True)
        stopping.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Serial startup: one instance at a time to avoid overloading model loading.
        # All instances for GPU 0 start before GPU 1, etc.
        for gid in gpu_ids:
            for inst in range(instances_per_gpu):
                if stopping.is_set():
                    break
                port = args.base_port + gid * instances_per_gpu + inst
                label = _make_label(gid, inst, instances_per_gpu)
                child = launch_child(
                    gpu_id=gid, instance_id=inst, port=port,
                    instances_per_gpu=instances_per_gpu,
                    config_path=args.config,
                    log_dir=log_dir, emit_lock=emit_lock,
                )
                children.append(child)
                if not _wait_ready(port, args.ready_timeout_s, label, emit_lock):
                    with emit_lock:
                        print(f"[{label}] failed to become ready. Aborting.", flush=True)
                    stopping.set()
                    break
                with emit_lock:
                    print(f"[{label}] READY  port={port}", flush=True)
            if stopping.is_set():
                break

        if not stopping.is_set():
            with emit_lock:
                ports_str = ", ".join(
                    str(args.base_port + g * instances_per_gpu + i)
                    for g in gpu_ids for i in range(instances_per_gpu)
                )
                print(f"[main] all {len(children)} workers READY; ports=[{ports_str}]",
                      flush=True)

            # Idle loop: wait for signal; auto-restart any child that dies.
            #
            # Counters per port (globally unique key, one entry per child instance):
            #   total_restarts[port]   — cumulative (diagnostic only)
            #   fail_streak[port]      — consecutive restart *failures*. Reset to 0
            #                           when a new child reaches READY and stays up
            #                           past flap_window_s. Partial increment (+0.5)
            #                           when it comes up but flaps within the window.
            #   last_ready_at[port]    — wall-clock time the current child became READY
            #
            # Using port as key ensures each (gpu_id, instance_id) pair has independent
            # counters — a crash on one instance does not affect sibling instances.
            # We only abort if fail_streak >= max_restart_fails for the SAME port.
            total_restarts: Dict[int, int] = {c.port: 0 for c in children}
            fail_streak: Dict[int, float] = {c.port: 0.0 for c in children}
            last_ready_at: Dict[int, float] = {c.port: time.time() for c in children}
            MAX_RESTART_FAILS = float(args.max_restart_fails)

            while not stopping.is_set():
                for i, c in enumerate(children):
                    label = _make_label(c.gpu_id, c.instance_id, instances_per_gpu)

                    # --- Hung-worker watchdog ---
                    # A native thread crash (e.g. libgomp "Thread creation failed"
                    # → SIGSEGV inside PyTorch/pycolmap) can leave the per-GPU
                    # threading.Lock permanently held while the uvicorn process
                    # itself stays alive (health → 200 OK).  popen.poll() never
                    # fires, so the normal dead-child restart path is bypassed and
                    # every subsequent request blocks forever until the client
                    # times out (1800s).  Detect this by tracking how long the
                    # oldest in-flight request has been waiting.
                    if args.hung_timeout_s > 0 and c.popen.poll() is None:
                        age = c.tailer.oldest_inflight_age()
                        if age is not None and age > args.hung_timeout_s:
                            with emit_lock:
                                print(f"[{label}] HUNG  oldest_inflight={age:.0f}s "
                                      f"> hung_timeout={args.hung_timeout_s:.0f}s — "
                                      f"force-killing", flush=True)
                            try:
                                c.popen.kill()
                            except Exception:
                                pass
                            # Let it fall through to the dead-child path below.
                            try:
                                c.popen.wait(timeout=5.0)
                            except subprocess.TimeoutExpired:
                                pass

                    if c.popen.poll() is None:
                        continue

                    rc = c.popen.returncode
                    port = c.port
                    uptime = time.time() - last_ready_at[port]
                    total_restarts[port] += 1
                    # Flap detection: if the previous incarnation crashed within
                    # flap_window_s of becoming READY, it's a crash-loop signal.
                    flapping = uptime < args.flap_window_s
                    if flapping:
                        fail_streak[port] += 0.5
                    with emit_lock:
                        print(f"[{label}] DIED rc={rc} uptime={uptime:.1f}s "
                              f"total_restarts={total_restarts[port]} "
                              f"fail_streak={fail_streak[port]:.1f} "
                              f"{'FLAPPING' if flapping else ''}", flush=True)
                    # Stop the old tailer before starting a fresh one on the
                    # truncated log file.
                    try:
                        c.tailer.stop()
                    except Exception:
                        pass
                    # Cooldown — gives OS time to reclaim fds / CUDA time to reset.
                    if args.restart_cooldown_s > 0:
                        time.sleep(args.restart_cooldown_s)
                    try:
                        new_child = launch_child(
                            gpu_id=c.gpu_id, instance_id=c.instance_id, port=port,
                            instances_per_gpu=instances_per_gpu,
                            config_path=args.config,
                            log_dir=log_dir, emit_lock=emit_lock,
                        )
                    except Exception as e:
                        fail_streak[port] += 1
                        with emit_lock:
                            print(f"[{label}] RESTART FAILED to spawn: {e} "
                                  f"(fail_streak={fail_streak[port]:.1f}/"
                                  f"{MAX_RESTART_FAILS})", flush=True)
                        if fail_streak[port] >= MAX_RESTART_FAILS:
                            with emit_lock:
                                print(f"[main] [{label}] port={port} has failed "
                                      f"{fail_streak[port]:.1f} restarts. "
                                      f"Aborting.", flush=True)
                            stopping.set()
                            break
                        continue
                    if _wait_ready(port, args.ready_timeout_s, label, emit_lock):
                        children[i] = new_child
                        last_ready_at[port] = time.time()
                        # Reset streak on a clean restart (even if prior was flapping).
                        fail_streak[port] = 0.0
                        with emit_lock:
                            print(f"[{label}] RESTARTED port={port} "
                                  f"pid={new_child.popen.pid} "
                                  f"total_restarts={total_restarts[port]}",
                                  flush=True)
                    else:
                        # Spawn succeeded but never became ready — tear down
                        # this child and count it as a full failure.
                        try:
                            if new_child.popen.poll() is None:
                                new_child.popen.terminate()
                            new_child.tailer.stop()
                        except Exception:
                            pass
                        fail_streak[port] += 1
                        with emit_lock:
                            print(f"[{label}] RESTART never became ready "
                                  f"(fail_streak={fail_streak[port]:.1f}/"
                                  f"{MAX_RESTART_FAILS})", flush=True)
                        if fail_streak[port] >= MAX_RESTART_FAILS:
                            with emit_lock:
                                print(f"[main] [{label}] port={port} has failed "
                                      f"{fail_streak[port]:.1f} restarts. "
                                      f"Aborting.", flush=True)
                            stopping.set()
                            break
                time.sleep(2.0)
    finally:
        shutdown(children, emit_lock)


if __name__ == "__main__":
    main()
