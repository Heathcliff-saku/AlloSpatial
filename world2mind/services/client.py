"""
HTTP client for the World2Mind Model Service.

Provides a clean interface for calling the cognitive map pipeline endpoint.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set

import requests

logger = logging.getLogger(__name__)


class ModelServiceClient:
    """
    HTTP client for the model service.

    Handles communication with the FastAPI model service,
    including retries and timeout management.
    """

    def __init__(self, service_url: str = "http://localhost:8100", timeout: int = 1800,
                 max_retries: int = 2, retry_delay: float = 5.0):
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _post(self, endpoint: str, payload: dict) -> dict:
        """Send POST request with retry logic."""
        url = f"{self.service_url}{endpoint}"
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json()
                if not result.get("success"):
                    raise RuntimeError(f"Service returned error: {result.get('message', 'unknown')}")
                return result
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"Connection failed (attempt {attempt + 1}), retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.error(f"Request timed out after {self.timeout}s")
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"Service request failed after {self.max_retries + 1} attempts: {last_error}")

    def health(self) -> dict:
        """Check service health."""
        resp = requests.get(f"{self.service_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def run_cognitive_map(
        self,
        image_dir: str,
        workspace: str,
        categories: List[str],
        scene_type: str = "outdoor",
        run_landmark: bool = True,
        run_route: bool = True,
        output_format: str = "grid",
        traversable_categories: Optional[List[str]] = None,
        config_path: str = "",
        scene_id: str = "",
    ) -> Dict[str, Any]:
        """
        Call the /cognitive_map endpoint to run the full pipeline.

        Returns:
            Dict with success, scene_id, output paths for yaml/visualizations.
        """
        payload = {
            "image_dir": image_dir,
            "workspace": workspace,
            "categories": categories,
            "scene_type": scene_type,
            "run_landmark": run_landmark,
            "run_route": run_route,
            "output_format": output_format,
            "traversable_categories": traversable_categories or [],
            "config_path": config_path,
            "scene_id": scene_id,
        }
        result = self._post("/cognitive_map", payload)
        return result.get("data", {})

    def is_available(self) -> bool:
        """Check if the service is reachable."""
        try:
            self.health()
            return True
        except Exception:
            return False
