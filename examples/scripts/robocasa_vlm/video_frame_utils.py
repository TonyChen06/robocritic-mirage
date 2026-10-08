"""
Utilities for on-the-fly video frame extraction and side-by-side overlay creation.

Supports two source types:
- MP4 files: decoded with `decord` (efficient random access)
- PNG directories: fallback for pre-extracted frames (frame_000000.png, etc.)
"""

import hashlib
import os
import tempfile
from collections import OrderedDict
from functools import lru_cache
import pdb
import numpy as np
from PIL import Image

_S3_VIDEO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "video_s3_cache")


# Reader cache keyed by (pid, path). decord's FFmpeg state is NOT fork-safe:
# accessing a VideoReader that was opened in a parent process and inherited by a
# forked child segfaults. Keying on os.getpid() guarantees every process (main,
# train workers, eval workers) opens its own reader and never touches one
# inherited across a fork. The inherited objects still sit in the child's copy
# of the dict but are never accessed, so they cause no harm.
#
# Bounded LRU: every distinct video path holds an open file descriptor inside
# its VideoReader. A training loop reuses a fixed set of paths, but a
# long-running server (e.g. rank_serve_robocasa.py) sees an unbounded stream of
# unique rollout videos, so an unbounded cache leaks fds until the process hits
# EMFILE ("Too many open files"). Capping the cache evicts the
# least-recently-used reader; dropping the last reference lets decord close its
# underlying fd(s) on __del__.
_VIDEO_READER_CACHE_MAX = 64
_video_reader_cache = OrderedDict()


def _get_video_reader(video_path: str):
    """Return a process-local decord.VideoReader, opening it on first use."""
    import decord
    key = (os.getpid(), video_path)
    vr = _video_reader_cache.get(key)
    if vr is None:
        vr = decord.VideoReader(video_path, num_threads=1)
        _video_reader_cache[key] = vr
        while len(_video_reader_cache) > _VIDEO_READER_CACHE_MAX:
            _, evicted = _video_reader_cache.popitem(last=False)
            del evicted  # drop ref so decord closes the underlying fd(s)
    else:
        _video_reader_cache.move_to_end(key)  # mark most-recently-used
    return vr


def extract_frame(source_path: str, frame_idx: int) -> Image.Image:
    if os.environ.get("ROBOCRITIC_MIRAGE", "0") != "1":
        return _extract_frame(source_path, frame_idx)
    frames = [np.asarray(_extract_frame(source_path, max(0, frame_idx - offset)).convert("RGB"),
                         dtype=np.float32) for offset in (16, 12, 8, 4, 0)]
    blended = sum(weight * frame for weight, frame in zip((.04, .08, .12, .16, .60), frames))
    return Image.fromarray(np.rint(blended).clip(0, 255).astype(np.uint8))


def _extract_frame(source_path: str, frame_idx: int) -> Image.Image:
    """
    Extract a single frame as a PIL Image.

    Args:
        source_path: Either an MP4 file path or a directory of PNGs.
        frame_idx: The frame index to extract.

    Returns:
        PIL Image in RGB mode.
    """
    if source_path.endswith(".mp4"):
        vr = _get_video_reader(source_path)
        frame = vr[frame_idx]
        # decord returns NDArray regardless of bridge; .asnumpy() always works
        frame_array = frame.asnumpy()
        return Image.fromarray(frame_array)
    else:
        png_path = os.path.join(source_path, f"frame_{frame_idx:06d}.png")
        if not os.path.isfile(png_path):
            png_path = os.path.join(source_path, f"frame_{frame_idx:06d}.jpg")
        return Image.open(png_path).convert("RGB")


def create_side_by_side(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Create a side-by-side image with a yellow separator line.

    Args:
        img1: Left image.
        img2: Right image (resized to match img1 if needed).

    Returns:
        Combined PIL Image (width*2 + 2, height).
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    w, h = img1.size
    combined = Image.new("RGB", (w * 2 + 2, h))
    combined.paste(img1, (0, 0))
    combined.paste(Image.new("RGB", (2, h), (255, 255, 0)), (w, 0))
    combined.paste(img2, (w + 2, 0))
    return combined


def _download_s3_video(s3_uri: str) -> str:
    """Download a video (or PNG directory) from S3 to a local cache directory.

    Returns the local path to the downloaded MP4 file or PNG directory.
    """
    os.makedirs(_S3_VIDEO_CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(s3_uri.encode()).hexdigest()

    # Try MP4 first
    if s3_uri.endswith(".mp4"):
        local_mp4 = os.path.join(_S3_VIDEO_CACHE_DIR, f"{cache_key}.mp4")
        if os.path.isfile(local_mp4):
            return local_mp4

        u = urlparse(s3_uri)
        bucket = u.netloc
        key = u.path.lstrip("/")

        if s3_key_exists(bucket, key):
            s3.download_file(bucket, key, local_mp4)
            return local_mp4

        # Try PNG directory (strip .mp4 from key)
        png_prefix = key[:-4] + "/"
        local_dir = os.path.join(_S3_VIDEO_CACHE_DIR, cache_key)
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            return local_dir

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=png_prefix)
        contents = resp.get("Contents", [])
        if contents:
            os.makedirs(local_dir, exist_ok=True)
            for obj in contents:
                fname = obj["Key"].split("/")[-1]
                s3.download_file(bucket, obj["Key"], os.path.join(local_dir, fname))
            return local_dir

    raise FileNotFoundError(f"Neither MP4 nor PNG directory found on S3 for {s3_uri}")


def get_frame_source(video_path: str) -> str:
    """
    Determine the best source for frames: MP4 file or PNG directory.

    Args:
        video_path: Path to the MP4 file (as stored in eval_log.json),
                    or an S3 URI (s3://bucket/key).

    Returns:
        A local path to the MP4 file or PNG directory.

    Raises:
        FileNotFoundError: If the source cannot be found locally or on S3.
    """
    if video_path.startswith("s3://"):
        return _download_s3_video(video_path)
    if os.path.isfile(video_path):
        return video_path
    png_dir = video_path[:-4]  # strip .mp4
    if os.path.isdir(png_dir):
        return png_dir
    raise FileNotFoundError(f"Neither {video_path} nor {png_dir} found")

import os
import glob
import boto3
from urllib.parse import urlparse

s3 = boto3.client("s3")

def parse_s3_uri(uri: str):
    # uri like s3://bucket/some/prefix
    u = urlparse(uri)
    bucket = u.netloc
    prefix = u.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix

def s3_list_immediate_subprefixes(bucket: str, prefix: str):
    """
    Lists "directories" immediately under prefix using Delimiter='/'
    Returns list of prefixes like 'some/prefix/job123/'
    """
    subprefixes = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix, Delimiter="/")
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)

        for cp in resp.get("CommonPrefixes", []):
            subprefixes.append(cp["Prefix"])

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return subprefixes

def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def read_s3_json(uri: str):
    """Read a JSON file from S3 and return the parsed object."""
    import json
    bucket, prefix = parse_s3_uri(uri)
    # parse_s3_uri adds trailing '/', strip it for a file key
    key = prefix.rstrip("/")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def find_job_dirs(task_path: str):
    if task_path.startswith("s3://"):
        bucket, prefix = parse_s3_uri(task_path)
        subdirs = s3_list_immediate_subprefixes(bucket, prefix)

        # Keep only those with eval_log.json
        job_prefixes = []
        for p in subdirs:
            if s3_key_exists(bucket, p + "eval_log.json"):
                job_prefixes.append(f"s3://{bucket}/{p}")
        return sorted(job_prefixes)

    # Local filesystem fallback
    all_dirs = sorted(glob.glob(f"{task_path}/*"))
    job_dirs = [
        d for d in all_dirs
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "eval_log.json"))
    ]
    return job_dirs
