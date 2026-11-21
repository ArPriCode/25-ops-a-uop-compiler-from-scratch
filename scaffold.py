"""
25 Ops: A UOp Compiler from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

from solution import *  # noqa: F401,F403

import time
import hashlib
import numpy as np

M32 = 0xFFFFFFFF

print("== Week 1: threefry2x32-20 with arithmetic only ==")
tf = parse_uir(threefry_program())
for key, ref in [((0, 0, 0, 0), (0x6B200159, 0x99BA4EFE)), ((M32,) * 4, (0x1CB996FC, 0xBB002BE7))]:
    got = tuple(int(v) & M32 for v in run_c(tf, dict(enumerate(key))))
    print("  key", [hex(k) for k in key], "->", [hex(v) for v in got], "matches" if got == ref else "MISMATCH")
print("  UOps:", count_uops(tf), "| ops:", " ".join(sorted({u.op for u in tf.toposort()})))

print("== Week 3: GEMM with explicit loops ==")
n = 128
rng = np.random.default_rng(0)
A, B = rng.standard_normal(n * n).astype(np.float32), rng.standard_normal(n * n).astype(np.float32)
gemm = parse_uir(gemm_program(n, n, n))
run_program(gemm, {0: A, 1: B})
t = time.time()
C = run_program(gemm, {0: A, 1: B})[2].reshape(n, n)
dt = time.time() - t
ref = np.zeros((n, n), np.float32)
for k in range(n):
    ref = (ref + A.reshape(n, n)[:, k:k + 1] * B.reshape(n, n)[k:k + 1, :]).astype(np.float32)
print(f"  {n}x{n}x{n}: bit-exact {np.array_equal(C, ref)}, {2 * n ** 3 / dt / 1e9:.2f} GFLOP/s (naive loops)")

print("== Week 5: Keccak-f[1600] ==")
kp = parse_uir(keccak_program())
for msg in (b"", b"abc"):
    block = bytearray(200)
    block[:len(msg)] = msg
    block[len(msg)] ^= 0x06
    block[135] ^= 0x80
    bits = np.unpackbits(np.frombuffer(bytes(block), np.uint8), bitorder="little").astype(np.int32)
    digest = np.packbits(run_uir(kp, {0: bits})[0].astype(np.uint8), bitorder="little").tobytes()[:32].hex()
    print(f"  SHA3-256({msg!r}) = {digest[:16]}... {'matches hashlib' if digest == hashlib.sha3_256(msg).hexdigest() else 'MISMATCH'}")
print("  UOps:", count_uops(kp), "(course budget: 99)")

print("== Week 6: MNIST trained by one compiled .uir program ==")
files = prepare_mnist(60000, 10000)
t = time.time()
res = run_uir(mnist_program(60000, 10000, files))
print(f"  784-32-10 MLP, 1 epoch of SGD, batch 16: test accuracy {int(res[5][0]) / 10000:.4f} in {time.time() - t:.1f} s")

