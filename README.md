# 25 Ops: A UOp Compiler from Scratch

Build the compiler from the Compilers for Machine Learning course one week at a time, in the course's own style: programs are hash-consed graphs of UOps written in a small text format. Week 1 parses and prints the format, infers dtypes, shapes and integer bounds, evaluates programs with exact i32 and f32 semantics, simplifies them with a bottom-up rewrite engine and renders them to C; the milestone is a threefry2x32 random number generator written with nothing but arithmetic. Week 2 adds buffers, loads, stores and ordered effects; week 3 adds loops and a GEMM; week 4 lowers shaped programs (REDUCE, EXPAND, STAGE) to scalar loops; week 5 adds reshape and permute, pad and flip, and runs Keccak-f[1600] (checked against SHA3-256) in under 99 UOps; week 6 adds GRADIENT and trains an MNIST classifier whose whole training loop is one compiled program.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** UOp
- [x] **2.** infer
- [x] **3.** parse_uir
- [ ] **4.** print_uir
- [ ] **5.** bounds
- [ ] **6.** eval_uop
- [ ] **7.** graph_rewrite
- [ ] **8.** fold_constants
- [ ] **9.** symbolic
- [ ] **10.** inline_calls
- [ ] **11.** render_c
- [ ] **12.** compile_c
- [ ] **13.** threefry_program
- [ ] **14.** verify_memory
- [ ] **15.** emit_c
- [ ] **16.** render_kernel
- [ ] **17.** check_scopes
- [ ] **18.** schedule
- [ ] **19.** run_program
- [ ] **20.** gemm_program
- [ ] **21.** lower_index
- [ ] **22.** rangeify
- [ ] **23.** run_uir
- [ ] **24.** movement_index
- [ ] **25.** pad_composite
- [ ] **26.** flip_composite
- [ ] **27.** keccak_program
- [ ] **28.** gradient
- [ ] **29.** expand_gradients
- [ ] **30.** fetch_buffer
- [ ] **31.** mnist_program

---

Built on Deep-ML.
