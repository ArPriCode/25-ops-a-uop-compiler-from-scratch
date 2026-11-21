"""
25 Ops: A UOp Compiler from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - UOp
import sys
import os
import re
import math
import itertools
import shutil
import subprocess
import tempfile
import hashlib
import ctypes
import pathlib
import urllib.request
from math import prod
from collections import namedtuple
import numpy as np


sys.setrecursionlimit(100000)

ParamArg = namedtuple("ParamArg", ["slot", "dtype", "lo", "hi"])
BufArg = namedtuple("BufArg", ["slot", "dtype", "shape", "url", "sha256"])
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def wrap32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _akey(a):
    if isinstance(a, tuple):
        return (type(a).__name__,) + tuple(_akey(x) for x in a)
    if isinstance(a, float):
        return ("float", "nan" if a != a else a.hex())
    return (type(a).__name__, a)


class UOp:
    _cache = {}
    __slots__ = ("op", "src", "arg", "_memo")

    def __new__(cls, op, src=(), arg=None):
        src = tuple(src)
        key = (op, src, _akey(arg))
        if key in cls._cache:
            return cls._cache[key]
        obj = object.__new__(cls)
        obj.op = op
        obj.src = src
        obj.arg = arg
        obj._memo = {}
        cls._cache[key] = obj
        return obj

    def __repr__(self):
        return f"UOp({self.op}, arg={self.arg!r}, src={len(self.src)})"

    def toposort(self):
        visited = set()
        order = []
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for s in reversed(node.src):
                if s not in visited:
                    stack.append((s, False))
        return order


def const(value, dtype=None):
    if dtype is None:
        if isinstance(value, (bool, np.bool_)):
            dtype = "bool"
        elif isinstance(value, (float, np.floating)):
            dtype = "f32"
        else:
            dtype = "i32"
    if dtype == "bool":
        if isinstance(value, (float, np.floating)):
            raise TypeError("a float cannot be a bool constant")
        v = bool(value)
    elif dtype == "i32":
        if isinstance(value, (bool, np.bool_, float, np.floating)):
            raise TypeError(f"{value!r} cannot be an i32 constant")
        v = wrap32(value)
    elif dtype == "f32":
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("a bool cannot be an f32 constant")
        v = float(np.float32(value))
    else:
        raise TypeError(f"unknown dtype {dtype!r}")
    return UOp("CONST", (), v)

# Step 2 - infer (not yet solved)
# TODO: implement

# Step 3 - parse_uir (not yet solved)
# TODO: implement

# Step 4 - print_uir (not yet solved)
# TODO: implement

# Step 5 - bounds (not yet solved)
# TODO: implement

# Step 6 - eval_uop (not yet solved)
# TODO: implement

# Step 7 - graph_rewrite (not yet solved)
# TODO: implement

# Step 8 - fold_constants (not yet solved)
# TODO: implement

# Step 9 - symbolic (not yet solved)
# TODO: implement

# Step 10 - inline_calls (not yet solved)
# TODO: implement

# Step 11 - render_c (not yet solved)
# TODO: implement

# Step 12 - compile_c (not yet solved)
# TODO: implement

# Step 13 - threefry_program (not yet solved)
# TODO: implement

# Step 14 - verify_memory (not yet solved)
# TODO: implement

# Step 15 - emit_c (not yet solved)
# TODO: implement

# Step 16 - render_kernel (not yet solved)
# TODO: implement

# Step 17 - check_scopes (not yet solved)
# TODO: implement

# Step 18 - schedule (not yet solved)
# TODO: implement

# Step 19 - run_program (not yet solved)
# TODO: implement

# Step 20 - gemm_program (not yet solved)
# TODO: implement

# Step 21 - lower_index (not yet solved)
# TODO: implement

# Step 22 - rangeify (not yet solved)
# TODO: implement

# Step 23 - run_uir (not yet solved)
# TODO: implement

# Step 24 - movement_index (not yet solved)
# TODO: implement

# Step 25 - pad_composite (not yet solved)
# TODO: implement

# Step 26 - flip_composite (not yet solved)
# TODO: implement

# Step 27 - keccak_program (not yet solved)
# TODO: implement

# Step 28 - gradient (not yet solved)
# TODO: implement

# Step 29 - expand_gradients (not yet solved)
# TODO: implement

# Step 30 - fetch_buffer (not yet solved)
# TODO: implement

# Step 31 - mnist_program (not yet solved)
# TODO: implement

