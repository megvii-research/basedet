#!/usr/bin/env python3

import functools
import os
from typing import Optional
import portalocker
from megfile import FSPath, S3Path, SmartPath, fs, s3

from basecore.utils import ensure_dir

__all__ = ["file_lock", "get_cache_dir"]


def patch_extract_protocol(f):

    @functools.wraps(f)
    def patched_f(path):

        if isinstance(path, str) and path.startswith("cache_s3"):
            protocol = "cache_s3"
            path_without_protocol = path[len(protocol) + 3:]
            return protocol, path_without_protocol
        else:
            return f(path)

    return patched_f


SmartPath._extract_protocol = patch_extract_protocol(SmartPath._extract_protocol)


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $BASEDET_CACHE, if set
        2) otherwise /data/.cache/basedet_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("BASEDET_CACHE", "/data/.cache/basedet_cache")
        )
    return cache_dir


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

    >>> filename = "/path/to/file"
    >>> with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    ensure_dir(dirname)
    return portalocker.Lock(path + ".lock", timeout=1800)  # type: ignore


def _bind_function(f):

    @functools.wraps(f)
    def cache_s3_method(path, *args, **kwargs):
        cache_dir = get_cache_dir()
        full_path = os.path.join(cache_dir, str(path))
        if not os.path.exists(full_path):
            with file_lock(full_path):
                s3.s3_download("s3://" + str(path), full_path)
        return f(FSPath(full_path), *args, **kwargs)

    return cache_s3_method


@SmartPath.register
class CacheS3Path(S3Path):

    protocol = "cache_s3"

    exists = _bind_function(fs.fs_exists)
    load = _bind_function(fs.fs_load_from)
    move = _bind_function(fs.fs_move)
    remove = _bind_function(fs.fs_remove)
    copy = _bind_function(fs.fs_copy)
    open = _bind_function(FSPath.open)
