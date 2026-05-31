"""
Frame extraction module.
Extracts frames from video files.

Primary backend: torchcodec (VideoDecoder) — decodes entirely in-process via
libavcodec, with no subprocess spawning and no fd/thread resource exhaustion
under concurrent load.

Fallback: ffmpeg subprocess — used only when torchcodec is unavailable or for
video-from-images generation (write direction).

The switch from subprocess.run(ffmpeg) to torchcodec eliminates the
"Resource temporarily unavailable" (EAGAIN) crash that occurred when 8+
concurrent GRPO generations each spawned independent ffmpeg processes.
"""

import subprocess
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from scipy import sparse

# ---------------------------------------------------------------------------
# Torchcodec availability check
# ---------------------------------------------------------------------------

try:
    import cv2
    from torchcodec.decoders import VideoDecoder as _VideoDecoder
    _TORCHCODEC_AVAILABLE = True
except ImportError:
    _TORCHCODEC_AVAILABLE = False

logger = logging.getLogger(__name__)


def _frame_tensor_to_bgr(frame_tensor) -> np.ndarray:
    """Convert a single decoded frame tensor to BGR numpy array for cv2.imwrite.

    Handles both tensor layouts produced by different torchcodec versions:
      - NCHW / CHW: torchcodec <= 0.5  →  shape (3, H, W), needs transpose
      - NHWC / HWC: torchcodec >= 0.10 →  shape (H, W, 3), use directly

    Input channel order is always RGB; output is BGR (cv2 convention).
    """
    arr = frame_tensor.cpu().numpy()  # uint8
    if arr.ndim == 3 and arr.shape[0] == 3:
        # CHW (C, H, W) → HWC
        arr = arr.transpose(1, 2, 0)
    # arr is now (H, W, 3) uint8 RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class DynamicMaskLoader:
    """Loader for dynamic mask files (sparse format)."""

    def __init__(self, dyn_masks_path: Path):
        """
        Initialize dynamic mask loader.

        Args:
            dyn_masks_path: Path to dyn_masks.npz file
        """
        self.dyn_masks_path = Path(dyn_masks_path)
        self._data = None
        self._shape = None
        self._num_frames = 0

    def load(self) -> bool:
        """Load the npz file. Returns True if successful."""
        if not self.dyn_masks_path.exists():
            logger.warning(f"Dynamic masks file not found: {self.dyn_masks_path}")
            return False

        try:
            self._data = np.load(self.dyn_masks_path)
            self._shape = tuple(self._data['shape'])

            # Count number of frames
            frame_keys = [k for k in self._data.keys() if k.endswith('_data') and k != 'shape']
            self._num_frames = len(frame_keys)

            logger.info(f"Loaded dynamic masks: {self._num_frames} frames, shape {self._shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dynamic masks: {e}")
            return False

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def get_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get dynamic mask for a specific frame.

        Args:
            frame_idx: Frame index (0-based, corresponds to indexes.txt first column)

        Returns:
            Binary mask (H, W) where 1 = dynamic pixel, or None if not available
        """
        if self._data is None:
            return None

        data_key = f'f_{frame_idx}_data'
        indices_key = f'f_{frame_idx}_indices'
        indptr_key = f'f_{frame_idx}_indptr'

        if data_key not in self._data:
            return None

        try:
            csr = sparse.csr_matrix(
                (self._data[data_key], self._data[indices_key], self._data[indptr_key]),
                shape=self._shape
            )
            return csr.toarray()
        except Exception as e:
            logger.warning(f"Failed to reconstruct mask for frame {frame_idx}: {e}")
            return None

    def calculate_dynamic_ratio(self) -> float:
        """
        Calculate the average ratio of dynamic pixels across all frames.

        Returns:
            Average ratio of dynamic pixels (0.0 to 1.0)
        """
        if self._data is None or self._shape is None:
            return 0.0

        total_pixels = self._shape[0] * self._shape[1]
        total_dynamic = 0
        frames_with_data = 0

        for i in range(self._num_frames):
            data_key = f'f_{i}_data'
            if data_key in self._data and len(self._data[data_key]) > 0:
                total_dynamic += len(self._data[data_key])
                frames_with_data += 1

        if frames_with_data == 0:
            return 0.0

        avg_dynamic_per_frame = total_dynamic / frames_with_data
        return avg_dynamic_per_frame / total_pixels


def load_frame_indexes(indexes_path: Path) -> List[Tuple[int, int]]:
    """
    Load frame index mapping from indexes.txt.

    Args:
        indexes_path: Path to indexes.txt file

    Returns:
        List of (mask_index, video_frame_index) tuples
    """
    indexes = []
    with open(indexes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                mask_idx = int(parts[0])
                video_frame_idx = int(parts[1])
                indexes.append((mask_idx, video_frame_idx))
    return indexes


def get_video_info(video_path: str) -> dict:
    """
    Get video information.

    Uses torchcodec VideoDecoder metadata when available (no subprocess),
    falls back to ffprobe otherwise.

    Returns:
        Dictionary with 'fps', 'total_frames', 'duration'
    """
    if _TORCHCODEC_AVAILABLE:
        return _get_video_info_torchcodec(video_path)
    return _get_video_info_ffprobe(video_path)


def _get_video_info_torchcodec(video_path: str) -> dict:
    """Get video metadata via torchcodec (no subprocess)."""
    decoder = _VideoDecoder(video_path)
    meta = decoder.metadata
    # average_fps may be None for variable-frame-rate videos; fall back to ffprobe
    video_fps = getattr(meta, 'average_fps', None) or getattr(meta, 'fps', None)
    if not video_fps:
        logger.warning(
            f"torchcodec could not determine fps for {video_path}, falling back to ffprobe")
        return _get_video_info_ffprobe(video_path)

    duration = getattr(meta, 'duration_seconds', None) or 0.0
    # num_frames may be None for container formats without frame count
    total_frames = getattr(meta, 'num_frames', None) or int(duration * video_fps)
    return {
        'fps': float(video_fps),
        'total_frames': int(total_frames),
        'duration': float(duration),
    }


def _get_video_info_ffprobe(video_path: str) -> dict:
    """Get video metadata via ffprobe subprocess (fallback)."""
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets,r_frame_rate',
        '-of', 'csv=p=0', video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    parts = result.stdout.strip().split(',')
    if len(parts) < 2:
        raise RuntimeError(f"Cannot parse video info: {result.stdout}")

    frame_rate_str = parts[0]
    total_packets = int(parts[1]) if parts[1] else 0

    if '/' in frame_rate_str:
        num, den = frame_rate_str.split('/')
        video_fps = float(num) / float(den)
    else:
        video_fps = float(frame_rate_str)

    duration = total_packets / video_fps if video_fps > 0 else 0
    return {
        'fps': video_fps,
        'total_frames': total_packets,
        'duration': duration,
    }


def extract_frames(
    video_path: str,
    output_dir: Path,
    fps: int = 5,
    max_frames: int = 150,
    image_quality: int = 2,
    image_format: str = "jpg"
) -> int:
    """
    Extract frames from video.

    Uses torchcodec when available (no subprocess, no fd exhaustion under
    concurrent load). Falls back to ffmpeg subprocess otherwise.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Target frames per second
        max_frames: Maximum number of frames to extract
        image_quality: ffmpeg -q:v parameter (1-31, lower is better).
                       When using torchcodec, mapped to JPEG quality (1→100, 31→1).
        image_format: Output image format (jpg, png)

    Returns:
        Number of extracted frames
    """
    if _TORCHCODEC_AVAILABLE:
        return _extract_frames_torchcodec(
            video_path, output_dir, fps, max_frames, image_quality, image_format)
    return _extract_frames_ffmpeg(
        video_path, output_dir, fps, max_frames, image_quality, image_format)


def _extract_frames_torchcodec(
    video_path: str,
    output_dir: Path,
    fps: int,
    max_frames: int,
    image_quality: int,
    image_format: str,
) -> int:
    """
    Extract frames via torchcodec VideoDecoder.

    Decodes video in-process (libavcodec), no subprocess spawning.
    Frames are decoded as (H, W, 3) uint8 RGB tensors and saved with cv2.
    """
    logger.info(f"Extracting frames from {video_path} (torchcodec)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Get video metadata ---
    video_info = get_video_info(video_path)
    video_fps = video_info['fps']
    video_duration = video_info['duration']
    total_frames = video_info['total_frames']

    # --- Calculate target frame indices ---
    # Mirror the ffmpeg fps-filter logic: pick uniformly-spaced frames.
    estimated_frames = int(video_duration * fps)
    if estimated_frames > max_frames:
        sample_interval = estimated_frames / max_frames
        actual_fps = fps / sample_interval
    else:
        actual_fps = fps

    logger.info(f"Video duration: {video_duration:.2f}s, target fps: {actual_fps:.2f}")

    if video_fps <= 0 or total_frames <= 0:
        raise RuntimeError(
            f"Invalid video metadata for {video_path}: fps={video_fps}, frames={total_frames}")

    # Frame step in source-video frames
    frame_step = max(1, int(round(video_fps / actual_fps)))
    frame_indices = list(range(0, total_frames, frame_step))[:max_frames]

    if not frame_indices:
        raise RuntimeError(f"No frames to extract from {video_path}")

    # --- Decode frames ---
    decoder = _VideoDecoder(video_path)
    frame_batch = decoder.get_frames_at(indices=frame_indices)
    # frame_batch.data: (N, H, W, 3) uint8, RGB

    # --- Map ffmpeg q:v [1,31] → cv2 JPEG quality [100,1] ---
    jpeg_quality = max(1, min(100, int(100 - (image_quality - 1) * (99 / 30))))
    encode_params: list = []
    if image_format in ("jpg", "jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif image_format == "png":
        # PNG compression 0-9 (0=no compression); map quality inverse
        png_compression = max(0, min(9, int((31 - image_quality) * 9 / 30)))
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    # --- Write frames to disk ---
    frames_data = frame_batch.data  # (N, C, H, W) or (N, H, W, C) depending on version
    frame_count = 0
    for i in range(frames_data.shape[0]):
        frame_bgr = _frame_tensor_to_bgr(frames_data[i])
        out_path = str(output_dir / f"frame_{i:06d}.{image_format}")
        ok = cv2.imwrite(out_path, frame_bgr, encode_params)
        if ok:
            frame_count += 1
        else:
            logger.warning(f"cv2.imwrite failed for frame {i}: {out_path}")

    logger.info(f"Extracted {frame_count} frames to {output_dir}")
    return frame_count


def _extract_frames_ffmpeg(
    video_path: str,
    output_dir: Path,
    fps: int,
    max_frames: int,
    image_quality: int,
    image_format: str,
) -> int:
    """Extract frames via ffmpeg subprocess (fallback when torchcodec unavailable)."""
    logger.info(f"Extracting frames from {video_path} (ffmpeg fallback)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(video_path)
    video_fps = video_info['fps']
    video_duration = video_info['duration']

    estimated_frames = int(video_duration * fps)
    if estimated_frames > max_frames:
        sample_interval = estimated_frames / max_frames
        actual_fps = fps / sample_interval
    else:
        actual_fps = fps

    logger.info(f"Video duration: {video_duration:.2f}s, target fps: {actual_fps:.2f}")

    output_pattern = str(output_dir / f"frame_%06d.{image_format}")
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={actual_fps}',
        '-q:v', str(image_quality),
        '-y',
        output_pattern
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True)
    if result.returncode != 0:
        stderr_text = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"ffmpeg failed: ...{stderr_text[-600:]}")

    frame_count = len(list(output_dir.glob(f"frame_*.{image_format}")))
    logger.info(f"Extracted {frame_count} frames to {output_dir}")
    return frame_count


def extract_frames_by_index(
    video_path: str,
    output_dir: Path,
    frame_indexes: List[Tuple[int, int]],
    image_quality: int = 2,
    image_format: str = "jpg"
) -> int:
    """
    Extract specific frames from video by frame index.
    Frame naming uses mask_index to align with dynamic masks.

    Uses torchcodec when available: a single decoder.get_frames_at() call
    replaces the original loop of N ffmpeg subprocesses (one per frame).

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_indexes: List of (mask_index, video_frame_index) tuples
        image_quality: ffmpeg -q:v parameter (1-31, lower is better)
        image_format: Output image format (jpg, png)

    Returns:
        Number of extracted frames
    """
    if _TORCHCODEC_AVAILABLE:
        return _extract_frames_by_index_torchcodec(
            video_path, output_dir, frame_indexes, image_quality, image_format)
    return _extract_frames_by_index_ffmpeg(
        video_path, output_dir, frame_indexes, image_quality, image_format)


def _extract_frames_by_index_torchcodec(
    video_path: str,
    output_dir: Path,
    frame_indexes: List[Tuple[int, int]],
    image_quality: int,
    image_format: str,
) -> int:
    """
    Batch-extract specific frames via torchcodec.

    Replaces N individual ffmpeg subprocess calls with a single
    decoder.get_frames_at(indices=[...]) batch decode.
    """
    logger.info(
        f"Extracting {len(frame_indexes)} indexed frames from {video_path} (torchcodec)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate mask indices from video frame indices
    mask_indices = [m for m, _ in frame_indexes]
    video_indices = [v for _, v in frame_indexes]

    # Single batch decode — O(1) decoder opens instead of O(N) subprocesses
    decoder = _VideoDecoder(video_path)
    frame_batch = decoder.get_frames_at(indices=video_indices)
    # frame_batch.data: (N, H, W, 3) uint8 RGB

    jpeg_quality = max(1, min(100, int(100 - (image_quality - 1) * (99 / 30))))
    encode_params: list = []
    if image_format in ("jpg", "jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif image_format == "png":
        png_compression = max(0, min(9, int((31 - image_quality) * 9 / 30)))
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    frames_data = frame_batch.data
    extracted_count = 0
    for i, mask_idx in enumerate(mask_indices):
        frame_bgr = _frame_tensor_to_bgr(frames_data[i])
        out_path = str(output_dir / f"frame_{mask_idx:06d}.{image_format}")
        ok = cv2.imwrite(out_path, frame_bgr, encode_params)
        if ok:
            extracted_count += 1
        else:
            logger.warning(
                f"cv2.imwrite failed for mask_idx={mask_idx} "
                f"(video_frame={video_indices[i]}): {out_path}")

    logger.info(f"Extracted {extracted_count} indexed frames to {output_dir}")
    return extracted_count


def _extract_frames_by_index_ffmpeg(
    video_path: str,
    output_dir: Path,
    frame_indexes: List[Tuple[int, int]],
    image_quality: int,
    image_format: str,
) -> int:
    """Fallback: extract indexed frames via individual ffmpeg subprocesses."""
    logger.info(
        f"Extracting {len(frame_indexes)} indexed frames from {video_path} (ffmpeg fallback)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(video_path)
    video_fps = video_info['fps']

    extracted_count = 0
    for mask_idx, video_frame_idx in frame_indexes:
        timestamp = video_frame_idx / video_fps
        output_file = output_dir / f"frame_{mask_idx:06d}.{image_format}"
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', str(image_quality),
            str(output_file)
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if result.returncode == 0 and output_file.exists():
            extracted_count += 1
        else:
            logger.warning(
                f"Failed to extract frame {mask_idx} (video frame {video_frame_idx})")

    logger.info(f"Extracted {extracted_count} indexed frames to {output_dir}")
    return extracted_count


def check_dynamic_mask_availability(
    scene_id: str,
    annot_base: str,
    indexes_filename: str = "indexes.txt",
    dyn_masks_filename: str = "dyn_masks.npz",
    dyn_pixel_threshold: float = 0.01
) -> Tuple[bool, Optional[Path], Optional[Path], float]:
    """
    Check if dynamic mask files are available and if dynamic content is significant.

    Args:
        scene_id: Scene identifier (video name/UUID)
        annot_base: Base path for annotation files
        indexes_filename: Name of frame index file
        dyn_masks_filename: Name of dynamic masks file
        dyn_pixel_threshold: Minimum ratio of dynamic pixels to consider significant

    Returns:
        Tuple of (use_indexed_extraction, indexes_path, dyn_masks_path, dynamic_ratio)
    """
    if not annot_base:
        return False, None, None, 0.0

    annot_dir = Path(annot_base) / scene_id
    indexes_path = annot_dir / indexes_filename
    dyn_masks_path = annot_dir / dyn_masks_filename

    if not indexes_path.exists():
        logger.info(f"Indexes file not found: {indexes_path}")
        return False, None, None, 0.0

    if not dyn_masks_path.exists():
        logger.info(f"Dynamic masks file not found: {dyn_masks_path}")
        return False, None, None, 0.0

    # Load and check dynamic mask ratio
    loader = DynamicMaskLoader(dyn_masks_path)
    if not loader.load():
        return False, None, None, 0.0

    dynamic_ratio = loader.calculate_dynamic_ratio()
    logger.info(f"Dynamic pixel ratio: {dynamic_ratio*100:.2f}%")

    use_indexed = dynamic_ratio >= dyn_pixel_threshold
    if use_indexed:
        logger.info(f"Dynamic content detected (>= {dyn_pixel_threshold*100:.1f}%), using indexed extraction")
    else:
        logger.info(f"Dynamic content below threshold, using standard fps extraction")

    return use_indexed, indexes_path, dyn_masks_path, dynamic_ratio


def generate_video_from_images(
    image_dir: Path,
    output_path: Path,
    fps: int = 10
) -> bool:
    """
    Generate video from a directory of images using ffmpeg.

    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Output video FPS

    Returns:
        True if successful, False otherwise
    """
    import glob

    image_files = sorted(glob.glob(str(image_dir / "*.jpg"))) + sorted(glob.glob(str(image_dir / "*.png")))
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return False

    logger.info(f"Generating video from {len(image_files)} images...")

    list_file = image_dir / "filelist.txt"
    with open(list_file, 'w') as f:
        for img_path in image_files:
            f.write(f"file '{img_path}'\n")
            f.write(f"duration {1.0/fps}\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-vf', f'fps={fps}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        list_file.unlink()
        logger.info(f"Video saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Video generation failed: {e.stderr}")
        if list_file.exists():
            list_file.unlink()
        return False
