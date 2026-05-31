"""
Multi-port HTTP client for the World2Mind service (training use only).

Used by GRPO rollout when the service has been launched with
`train_multiproc_service.py`, which spawns N independent single-GPU processes
on consecutive ports (base_port, base_port+1, ..., base_port+N-1).

Dispatch policy (v2):
  1. Probe /health with a 5-second TTL cache (avoid hammering all ports on every call).
  2. Among reachable ports, pick the one with the fewest in-flight requests
     (client-side counter, updated atomically under _pick_lock).
     This eliminates the race where all concurrent callers see the same stale
     busy=False flag and pile onto the same GPU.
  3. Failover: if the chosen port raises an OSError/RuntimeError (crash, restart
     window), try the next-best reachable port, up to min(3, N) attempts total.

Bug fixes vs v1:
  - v1 _pick_client() probed /health on EVERY call (6 HTTP requests × N concurrent
    callers = thundering herd) and used a racy round-robin counter that let all
    concurrent callers select the same GPU simultaneously.
  - v1 had no failover: a crashed port raised RuntimeError immediately with no
    retry on other ports.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from services.client import ModelServiceClient

logger = logging.getLogger(__name__)

# TTL for /health probe cache.  Within this window, all callers share the same
# liveness snapshot.  5 s is a reasonable balance: short enough to detect a
# newly-restarted port, long enough to avoid per-call HTTP overhead.
_PROBE_CACHE_TTL = 5.0


class MultiPortModelClient:
    """Same public surface as ModelServiceClient, but load-balances across N ports.

    Port assignment mirrors train_multiproc_service.py:
        port = base_port + gpu_id * instances_per_gpu + inst_idx

    instances_per_gpu=1 (default): port = base_port + gpu_id — identical to
    the original single-instance-per-GPU behavior.
    """

    def __init__(
        self,
        base_port: int,
        gpu_ids: List[int],
        instances_per_gpu: int = 1,
        host: str = "http://localhost",
        timeout: int = 1800,
        health_timeout: float = 2.0,
    ):
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        instances_per_gpu = max(1, instances_per_gpu)
        self._host = host.rstrip('/')
        self._base_port = base_port
        self._gpu_ids = list(gpu_ids)
        self._instances_per_gpu = instances_per_gpu
        self._health_timeout = health_timeout
        self._clients: List[ModelServiceClient] = [
            ModelServiceClient(
                service_url=f"{self._host}:{base_port + gid * instances_per_gpu + inst}",
                timeout=timeout,
            )
            for gid in self._gpu_ids
            for inst in range(instances_per_gpu)
        ]
        n = len(self._clients)

        # In-flight counter: one entry per client, incremented before dispatching
        # and decremented in the finally block.  Protected by _pick_lock.
        self._inflight: List[int] = [0] * n
        self._pick_lock = threading.Lock()

        # /health probe cache: avoid N×M HTTP round-trips under concurrent load.
        self._probe_cache: List[Optional[bool]] = [None] * n
        self._probe_ts: float = 0.0
        self._probe_lock = threading.Lock()

        # Retained for backward compat (was public in v1, though unused externally)
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Health probing
    # ------------------------------------------------------------------

    def _probe_busy(self) -> List[Optional[bool]]:
        """Return one entry per client: True=busy, False=free, None=unreachable.

        Makes one HTTP GET /health per port; results are NOT cached here.
        Use _probe_busy_cached() for the cached wrapper.
        """
        busy_flags: List[Optional[bool]] = [None] * len(self._clients)
        for i, c in enumerate(self._clients):
            try:
                resp = requests.get(
                    f"{c.service_url}/health", timeout=self._health_timeout)
                resp.raise_for_status()
                data = resp.json()
                workers = data.get("workers") or []
                if workers and isinstance(workers[0], dict):
                    busy_flags[i] = bool(workers[0].get("busy", False))
                else:
                    busy_flags[i] = False
            except Exception:
                busy_flags[i] = None
        return busy_flags

    def _probe_busy_cached(self) -> List[Optional[bool]]:
        """Cached /health probe with TTL=5s.

        All concurrent callers within the TTL window share the same snapshot,
        eliminating the thundering-herd of N HTTP requests × M callers.
        """
        now = time.monotonic()
        # Fast-path: cache still warm (no lock needed for read — worst case we
        # do an extra probe, which is harmless).
        if now - self._probe_ts < _PROBE_CACHE_TTL:
            return self._probe_cache
        with self._probe_lock:
            # Re-check under lock (another thread may have refreshed already).
            if now - self._probe_ts < _PROBE_CACHE_TTL:
                return self._probe_cache
            result = self._probe_busy()
            self._probe_cache = result
            self._probe_ts = now
            return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _pick_client(self, exclude: Set[int] = None) -> Tuple[int, ModelServiceClient]:
        """Pick the client with the fewest in-flight requests.

        Args:
            exclude: Set of client indices to skip (used by failover).

        Returns:
            (index, client) — inflight[index] has already been incremented.
        """
        exclude = exclude or set()
        busy = self._probe_busy_cached()

        # Prefer reachable (not None) ports outside the exclude set.
        reachable = [
            i for i, b in enumerate(busy)
            if b is not None and i not in exclude
        ]
        if not reachable:
            # All probed as unreachable — widen to any non-excluded port.
            reachable = [i for i in range(len(self._clients)) if i not in exclude]
        if not reachable:
            # Absolute fallback: use everything (should never happen).
            reachable = list(range(len(self._clients)))

        with self._pick_lock:
            idx = min(reachable, key=lambda i: self._inflight[i])
            self._inflight[idx] += 1

        logger.debug(
            f"MultiPortClient: picked port idx={idx} "
            f"inflight={self._inflight[idx]} "
            f"(all={list(self._inflight)})"
        )
        return idx, self._clients[idx]

    # ------------------------------------------------------------------
    # Public API (mirrors ModelServiceClient)
    # ------------------------------------------------------------------

    def run_cognitive_map(self, *args, **kwargs) -> Dict[str, Any]:
        """Dispatch a cognitive_map request with automatic failover.

        Tries up to min(3, N) different ports.  On OSError/RuntimeError from a
        port (crashed, restarting), that port is added to the exclude set and
        the next-best available port is tried instead.
        """
        tried: Set[int] = set()
        last_error: Exception = RuntimeError("No clients available")
        max_attempts = min(3, len(self._clients))

        for attempt in range(max_attempts):
            idx, client = self._pick_client(exclude=tried)
            try:
                return client.run_cognitive_map(*args, **kwargs)
            except (OSError, RuntimeError, requests.ConnectionError,
                    requests.Timeout) as e:
                logger.warning(
                    f"MultiPortClient: attempt {attempt + 1}/{max_attempts} "
                    f"failed on port idx={idx} ({client.service_url}): {e}"
                )
                last_error = e
                tried.add(idx)
                # Mark this port as unreachable in the probe cache so
                # concurrent callers also avoid it during the TTL window.
                with self._probe_lock:
                    if idx < len(self._probe_cache):
                        self._probe_cache[idx] = None
            finally:
                # Always decrement, even on failure, so inflight never leaks.
                with self._pick_lock:
                    self._inflight[idx] = max(0, self._inflight[idx] - 1)

        raise RuntimeError(
            f"MultiPortClient: all {max_attempts} attempted ports failed. "
            f"Last error: {last_error}"
        )

    def health(self) -> Dict[str, Any]:
        """Aggregate health of all backend ports."""
        return {
            "workers": [
                {
                    "url": c.service_url,
                    "inflight": self._inflight[i],
                    **(_safe_health(c, self._health_timeout) or {"error": "unreachable"}),
                }
                for i, c in enumerate(self._clients)
            ]
        }

    def is_available(self) -> bool:
        """Return True iff at least one backend responds to /health."""
        for c in self._clients:
            if _safe_health(c, self._health_timeout) is not None:
                return True
        return False


def _safe_health(client: ModelServiceClient, timeout: float) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{client.service_url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None
