"""Upload report files to S3-compatible object storage (per-project bucket)."""
from __future__ import annotations

import os
from pathlib import Path

import boto3

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _resolve(config: dict, key: str) -> str:
    """Resolve a config value: if key_env exists, read from env; otherwise use key directly."""
    env_key = config.get(f"{key}_env")
    if env_key:
        return os.environ.get(env_key, "")
    return config.get(key, "")


def upload_report_files(
    task_id: str,
    local_dir: Path,
    files: list[dict],
    storage_config: dict,
) -> list[dict]:
    """Upload files to the project's bucket and return updated file list with public URLs.

    storage_config keys (direct or _env suffix for env var lookup):
        endpoint / endpoint_env: S3 endpoint
        access_key_id / access_key_id_env: S3 access key
        secret_access_key / secret_access_key_env: S3 secret key
        bucket: bucket name
        public_url: public base URL for the bucket
        prefix: optional key prefix (default: "deep-research")
    """
    endpoint = _resolve(storage_config, "endpoint")
    access_key = _resolve(storage_config, "access_key_id")
    secret_key = _resolve(storage_config, "secret_access_key")
    bucket = storage_config.get("bucket", "")
    public_url = storage_config.get("public_url", "").rstrip("/")
    prefix = storage_config.get("prefix", "deep-research")

    if not all([endpoint, bucket, public_url, access_key, secret_key]):
        logger.warning("storage_config_incomplete", bucket=bucket, has_endpoint=bool(endpoint))
        return files

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    updated = []
    for f in files:
        name = f["name"]
        local_path = local_dir / name
        if not local_path.exists():
            updated.append(f)
            continue

        key = f"{prefix}/{task_id}/{name}"
        content_type = f.get("type", "application/octet-stream")

        try:
            s3.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            public_file_url = f"{public_url}/{key}"
            updated.append({**f, "url": public_file_url})
            logger.info("file_uploaded", task_id=task_id, key=key)
        except Exception as e:
            logger.error("upload_failed", task_id=task_id, file=name, error=str(e))
            updated.append(f)  # keep original local URL on failure

    return updated
