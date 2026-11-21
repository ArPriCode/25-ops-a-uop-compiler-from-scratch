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

# Step 2 - infer
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


NP_DTYPE = {"bool": np.bool_, "i32": np.int32, "f32": np.float32}
ALU_OPS = ("ADD", "MUL", "MAX", "XOR", "CMPLT", "SHL", "DIV", "MOD", "WHERE", "CAST", "RECIP", "TRUNC")
TYPE_RULES = {}


def infer(u):
    if "type" in u._memo:
        return u._memo["type"]
    if u.op not in TYPE_RULES:
        raise TypeError(f"no type rule for op {u.op!r}")
    res = TYPE_RULES[u.op](u)
    u._memo["type"] = res
    return res


UOp.dtype = property(lambda self: infer(self)[0])
UOp.shape = property(lambda self: infer(self)[1])


def _same(srcs, op):
    d, s = srcs[0].dtype, srcs[0].shape
    for x in srcs[1:]:
        if x.dtype != d or x.shape != s:
            raise TypeError(f"{op}: operand {x.dtype}{list(x.shape)} does not match {d}{list(s)}")
    return d, s


def _alu_rule(allowed, out=None):
    def rule(u):
        if len(u.src) != 2:
            raise TypeError(f"{u.op} takes two operands")
        d, s = _same(u.src, u.op)
        if d not in allowed:
            raise TypeError(f"{u.op} does not accept {d}")
        return (out or d), s
    return rule


def _unary_rule(allowed):
    def rule(u):
        if len(u.src) != 1 or u.src[0].dtype not in allowed:
            raise TypeError(f"{u.op} needs one {'/'.join(allowed)} operand")
        return u.src[0].dtype, u.src[0].shape
    return rule


def _scalar_i32(x, what):
    if x.dtype != "i32" or x.shape != ():
        raise TypeError(f"{what} must be an i32 scalar")


def _body_params(body):
    found, seen, stack = {}, set(), [body]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u.op == "PARAM":
            if u.arg.slot in found:
                raise TypeError(f"two PARAMs with slot {u.arg.slot} in one function body")
            found[u.arg.slot] = u
        stack.extend(u.src[1:] if u.op == "CALL" else u.src)
    return found


def _call_rule(u):
    body, args = u.src[0], u.src[1:]
    for slot, p in _body_params(body).items():
        if slot >= len(args):
            raise TypeError(f"CALL is missing argument {slot}")
        if args[slot].dtype != p.arg.dtype or args[slot].shape != ():
            raise TypeError(f"CALL argument {slot} must be a {p.arg.dtype} scalar")
    return body.dtype, body.shape


def _stack_rule(u):
    if len(u.src) == 0:
        raise TypeError("STACK requires at least one source")
    d, s = _same(u.src, "STACK")
    return d, (len(u.src),) + s


def _index_rule(u):
    if len(u.src) < 2:
        raise TypeError("INDEX requires a tensor and at least one index")
    x, idxs = u.src[0], u.src[1:]
    if len(idxs) > len(x.shape):
        raise TypeError(f"INDEX has {len(idxs)} indices for rank {len(x.shape)} value")
    for idx in idxs:
        _scalar_i32(idx, "INDEX index")
    return x.dtype, x.shape[len(idxs):]


def _shrink_rule(u):
    if len(u.src) < 2:
        raise TypeError("SHRINK requires a tensor and at least one begin offset")
    x, begins = u.src[0], u.src[1:]
    sizes = u.arg
    if not isinstance(sizes, tuple) or len(sizes) != len(begins):
        raise TypeError("SHRINK sizes must match number of begin offsets")
    k = len(sizes)
    if k < 1 or k > len(x.shape):
        raise TypeError(f"SHRINK cannot slice {k} axes of rank {len(x.shape)} value")
    for b in begins:
        _scalar_i32(b, "SHRINK begin")
    for n, axis in zip(sizes, x.shape[:k]):
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n > axis:
            raise TypeError(f"SHRINK window size {n} out of bounds for axis {axis}")
    return x.dtype, tuple(sizes) + x.shape[k:]


def _where_rule(u):
    c, a, b = u.src
    if c.dtype != "bool":
        raise TypeError("WHERE condition must be bool")
    d, s = _same((a, b), "WHERE")
    if c.shape != s:
        raise TypeError("WHERE condition shape must match its values")
    return d, s


def _cast_rule(u):
    if u.arg not in NP_DTYPE:
        raise TypeError(f"cannot CAST to {u.arg!r}")
    return u.arg, u.src[0].shape


TYPE_RULES.update({
    "CONST": lambda u: ("bool" if isinstance(u.arg, bool) else "f32" if isinstance(u.arg, float) else "i32", ()),
    "PARAM": lambda u: (u.arg.dtype, ()),
    "CALL": _call_rule,
    "STACK": _stack_rule,
    "INDEX": _index_rule,
    "SHRINK": _shrink_rule,
    "ADD": _alu_rule(("i32", "f32")),
    "MUL": _alu_rule(("i32", "f32")),
    "MAX": _alu_rule(("bool", "i32", "f32")),
    "XOR": _alu_rule(("bool", "i32")),
    "CMPLT": _alu_rule(("bool", "i32", "f32"), out="bool"),
    "SHL": _alu_rule(("i32",)),
    "DIV": _alu_rule(("i32",)),
    "MOD": _alu_rule(("i32",)),
    "WHERE": _where_rule,
    "CAST": _cast_rule,
    "RECIP": _unary_rule(("f32",)),
    "TRUNC": _unary_rule(("f32",)),
})

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

