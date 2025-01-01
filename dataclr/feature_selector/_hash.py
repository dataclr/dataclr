from __future__ import annotations

from dataclr.methods.method import Method


def hash_set(obj_set: set) -> int:
    return hash(tuple(sorted(obj_set, key=hash)))


def get_combination_hash(method_set: set[Method], feature_list: list[str]) -> int:
    method_set_hash = hash_set(method_set)
    feature_list_hash = hash_set(set(feature_list))
    return hash((method_set_hash, feature_list_hash))
