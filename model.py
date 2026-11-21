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

# Step 3 - parse_uir
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


_TOKEN = re.compile(r"\(|\)|\[|\]|\{|\}|[^\s()\[\]{}]+")
_INT = re.compile(r"-?(0x[0-9a-f]+|\d+)$")
_FLOAT = re.compile(r"-?((\d+\.\d*|\.\d+|\d+)(e[-+]?\d+)?|inf|nan)$")
_ALIASES = {"floordiv": "DIV", "floormod": "MOD"}
_TEXT_OPS = {"const", "param", "call", "stack", "index", "shrink", "cast", "recip", "trunc", "add", "mul", "max",
             "xor", "cmplt", "shl", "div", "mod", "where", "floordiv", "floormod", "buffer", "alloc", "load",
             "store", "after", "range", "end", "reduce", "expand", "stage", "reshape", "permute", "gradient",
             "detach", "pad"}
_STRING_ATTRS = ("url", "sha256", "name")
_AUTO_SLOT = itertools.count(1 << 20)


class Lit:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


def _literal(tok):
    t = tok.lower()
    if t in ("true", "false"):
        return t == "true"
    if _INT.match(t):
        return int(t, 0) if "0x" in t else int(t)
    if _FLOAT.match(t) and any(c in t for c in ".einfa"):
        return float(t)
    return None


def _lit_dtype(v):
    return "bool" if isinstance(v, bool) else "f32" if isinstance(v, float) else "i32"


def _to_uop(x, dtype=None, shape=()):
    if isinstance(x, UOp):
        return x
    if isinstance(x, list):
        d, s = _data(x)
        return UOp("STACK", [_to_uop(e, d, s) for e in x])
    v, d = x.v, dtype or _lit_dtype(x.v)
    if d == "bool" and not isinstance(v, bool):
        if isinstance(v, float):
            raise TypeError("a float literal cannot be a bool")
        v = bool(v)
    if d in ("i32", "f32") and isinstance(v, bool):
        raise TypeError("a bool literal cannot stand next to a number")
    c = const(v, d)
    if shape:
        c = UOp("EXPAND", (UOp("RESHAPE", (c,), (1,) * len(shape)),), tuple(shape))
    return c


def _data(operands):
    for a in operands:
        if isinstance(a, UOp):
            return a.dtype, a.shape
    lits = [a.v for a in operands if isinstance(a, Lit)]
    for a in operands:
        if isinstance(a, list):
            return _data(a)[0], (len(a),)
    if any(isinstance(v, float) for v in lits):
        return "f32", ()
    if lits and all(isinstance(v, bool) for v in lits):
        return "bool", ()
    return "i32", ()


def _ints(x):
    if isinstance(x, Lit):
        return (int(x.v),)
    if isinstance(x, UOp) and x.op == "CONST":
        return (int(x.arg),)
    if isinstance(x, list):
        return tuple(i for e in x for i in _ints(e))
    raise TypeError("expected a list of integers")


def _window(op, args):
    """SHRINK/PAD operands -> (x, begins, sizes). Forms: x [[b e]...] | x b size | x b0 b1 ... [s0 s1 ...]."""
    x = args[0]
    if len(args) == 2 and isinstance(args[1], list) and args[1] and isinstance(args[1][0], list):
        pairs = [(int(p[0].v), int(p[1].v)) for p in args[1]]
        begins = [const(b) for b, _ in pairs]
        if op == "SHRINK":
            sizes = tuple(e - b for b, e in pairs)
        else:
            sizes = tuple(b + n + a for (b, a), n in zip(pairs, x.shape))
        return x, begins, sizes
    if isinstance(args[-1], list):
        return x, [_to_uop(b, "i32") for b in args[1:-1]], _ints(args[-1])
    if len(args) == 3:
        return x, [_to_uop(args[1], "i32")], _ints(args[2])
    raise SyntaxError(f"bad {op.lower()} operands")


def _build(op, args, attrs):
    OP = _ALIASES.get(op, op.upper())

    def need(n):
        if len(args) != n:
            raise SyntaxError(f"{op} takes {n} operand(s), got {len(args)}")

    if OP in ("ADD", "MUL", "MAX", "XOR", "CMPLT", "SHL", "DIV", "MOD"):
        need(2)
        d, s = _data(args)
        return UOp(OP, [_to_uop(a, d, s) for a in args])
    if OP == "WHERE":
        need(3)
        d, s = _data(args[1:])
        if s == () and isinstance(args[0], UOp):
            s = args[0].shape
        return UOp("WHERE", [_to_uop(args[0], "bool", s), _to_uop(args[1], d, s), _to_uop(args[2], d, s)])
    if OP in ("RECIP", "TRUNC"):
        need(1)
        return UOp(OP, [_to_uop(args[0], "f32")])
    if OP == "CAST":
        need(1)
        return UOp("CAST", [_to_uop(args[0])], attrs["dtype"])
    if OP == "CONST":
        need(1)
        return _to_uop(args[0], attrs.get("dtype"))
    if OP == "PARAM":
        if args and _ints(args[0]) != ():
            raise TypeError("PARAMs are scalars")
        lo, hi = attrs.get("min"), attrs.get("max")
        return UOp("PARAM", (), ParamArg(int(attrs["slot"]), attrs["dtype"], lo, hi))
    if OP == "CALL":
        fn, rest = args[0], args[1:]
        params = _body_params(fn)
        return UOp("CALL", [fn] + [_to_uop(a, params[i].arg.dtype if i in params else None) for i, a in enumerate(rest)])
    if OP == "STACK":
        return _to_uop(list(args))
    if OP == "INDEX":
        return UOp("INDEX", [_to_uop(args[0])] + [_to_uop(a, "i32") for a in args[1:]])
    if OP in ("SHRINK", "PAD"):
        x, begins, sizes = _window(OP, args)
        return UOp(OP, [_to_uop(x)] + begins, tuple(sizes))
    if OP in ("BUFFER", "ALLOC"):
        need(1)
        return UOp(OP, (), BufArg(int(attrs["slot"]), attrs["dtype"], _ints(args[0]), attrs.get("url"), attrs.get("sha256")))
    if OP == "LOAD":
        need(1)
        return UOp("LOAD", [_to_uop(args[0])])
    if OP == "STORE":
        need(2)
        dst = _to_uop(args[0])
        return UOp("STORE", [dst, _to_uop(args[1], dst.dtype, dst.shape)])
    if OP == "AFTER":
        return UOp("AFTER", [_to_uop(a) for a in args])
    if OP == "RANGE":
        need(1)
        slot = attrs.get("slot", attrs.get("name"))
        return UOp("RANGE", [const(_ints(args[0])[0])], next(_AUTO_SLOT) if slot is None else int(slot))
    if OP == "END":
        need(2)
        return UOp("END", [_to_uop(a) for a in args])
    if OP == "REDUCE":
        need(1)
        return UOp("REDUCE", [_to_uop(args[0])], (str(attrs.get("op", "add")).upper(), int(attrs.get("pop", 1))))
    if OP in ("EXPAND", "RESHAPE", "PERMUTE"):
        need(2)
        return UOp(OP, [_to_uop(args[0])], _ints(args[1]))
    if OP in ("STAGE", "DETACH"):
        need(1)
        return UOp(OP, [_to_uop(args[0])])
    if OP == "GRADIENT":
        need(2)
        return UOp("GRADIENT", [_to_uop(a) for a in args])
    raise SyntaxError(f"unknown op {op!r}")


def _lookup(name, scopes):
    for sc in reversed(scopes):
        if name in sc:
            return sc[name]
    raise NameError(f"undefined name {name}")


def _parse_operand(toks, pos, scopes):
    if pos >= len(toks):
        raise SyntaxError("unexpected end of expression")
    tok = toks[pos]
    if tok == "(":
        val, pos = _parse_expr(toks, pos + 1, scopes)
        if pos >= len(toks) or toks[pos] != ")":
            raise SyntaxError("expected ')'")
        return val, pos + 1
    if tok == "[":
        items = []
        pos += 1
        while pos < len(toks) and toks[pos] != "]":
            item, pos = _parse_operand(toks, pos, scopes)
            items.append(item)
        if pos >= len(toks) or toks[pos] != "]":
            raise SyntaxError("expected ']'")
        return items, pos + 1
    if tok.startswith("%"):
        if len(tok) == 1:
            raise SyntaxError("empty '%' name")
        return _lookup(tok[1:], scopes), pos + 1
    lit = _literal(tok)
    if lit is not None:
        return Lit(lit), pos + 1
    raise SyntaxError(f"unexpected token {tok!r}")


def _parse_expr(toks, pos, scopes):
    if pos >= len(toks):
        raise SyntaxError("unexpected end of expression")
    if toks[pos].lower() not in _TEXT_OPS:
        return _parse_operand(toks, pos, scopes)
    op = toks[pos].lower()
    pos += 1
    args = []
    attrs = {}
    after_colon = False
    while pos < len(toks) and toks[pos] not in (")", "]"):
        tok = toks[pos]
        if tok == ":":
            after_colon = True
            pos += 1
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.lower()
            if k in _STRING_ATTRS:
                attrs[k] = v
            else:
                lit = _literal(v)
                attrs[k] = lit if lit is not None else v.lower()
            pos += 1
            continue
        if after_colon and tok.lower() in NP_DTYPE:
            attrs["dtype"] = tok.lower()
            pos += 1
            continue
        val, pos = _parse_operand(toks, pos, scopes)
        args.append(val)
    return _build(op, args, attrs), pos


def parse_uir(text):
    scopes = [{}]
    block_stack = []
    last_val = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        toks = _TOKEN.findall(line)
        if not toks:
            continue

        if toks == ["}"]:
            if not block_stack:
                raise SyntaxError("unmatched '}'")
            fn_name, prev_last = block_stack.pop()
            finished_scope = scopes.pop()
            if "result" in finished_scope:
                fn_val = _to_uop(finished_scope["result"])
            elif last_val is not None:
                fn_val = _to_uop(last_val)
            else:
                raise SyntaxError(f"empty block for %{fn_name}")
            scopes[-1][fn_name] = fn_val
            last_val = fn_val
            continue

        if len(toks) >= 2 and toks[0].startswith("%") and toks[1] == "=":
            if len(toks[0]) == 1:
                raise SyntaxError("empty '%' name")
            name = toks[0][1:]
            if toks[2:] == ["{"]:
                block_stack.append((name, last_val))
                scopes.append({})
                last_val = None
                continue
            if len(toks) < 3:
                raise SyntaxError(f"missing expression after '%{name} ='")
            val, pos = _parse_expr(toks, 2, scopes)
            if pos != len(toks):
                raise SyntaxError(f"unexpected trailing tokens: {toks[pos:]}")
            val = _to_uop(val)
            scopes[-1][name] = val
            last_val = val
            continue

        raise SyntaxError(f"invalid statement: {line!r}")

    if block_stack:
        raise SyntaxError(f"unclosed block for %{block_stack[-1][0]}")

    root = scopes[0]
    if "result" in root:
        return root["result"]
    if last_val is None:
        raise SyntaxError("empty program")
    return last_val

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

