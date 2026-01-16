"""
Tests for cache safety behavior.
"""

from __future__ import annotations

import time

import pytest # type: ignore

from wordcloud.utils.performance_optimizations import DiskCache, LRUCache


def test_lru_cache_eviction() -> None:
    cache = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_ttl_expiration() -> None:
    cache = LRUCache(max_size=2, max_age_seconds=0.01)
    cache.put("a", 1)
    time.sleep(0.02)

    assert cache.get("a") is None


def test_disk_cache_cleanup_max_entries(tmp_path) -> None:
    cache = DiskCache(cache_dir=tmp_path, max_entries=1, cleanup_interval_seconds=0)
    cache.put("a", {"value": 1})
    cache.put("b", {"value": 2})

    cache_files = list(tmp_path.glob("*.cache"))
    assert len(cache_files) == 1


def test_disk_cache_cleanup_max_age(tmp_path) -> None:
    cache = DiskCache(cache_dir=tmp_path, max_age_seconds=0.01, cleanup_interval_seconds=0)
    cache.put("a", {"value": 1})
    time.sleep(0.02)
    cache.cleanup()

    cache_files = list(tmp_path.glob("*.cache"))
    assert len(cache_files) == 0
