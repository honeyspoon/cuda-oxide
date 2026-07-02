# LTOIR Typed-Pointer Limitation: Investigation

## Status: Investigation Complete

**Date:** 2026-07-02
**Scope:** Read-only investigation of the opaque-to-typed pointer gap in
cuda-oxide's LLVM IR export for pre-Blackwell NVVM targets.

---

## 1. Current State

cuda-oxide compiles Rust GPU kernels through a pipeline that ends with textual
LLVM IR.  When targeting NVVM (for libNVVM / nvJitLink), the exporter must
produce IR that libNVVM can parse.  Two NVVM "input dialects" exist:

| Dialect | Targets | Pointer syntax | Data layout |
|---------|---------|---------------|-------------|
| `NvvmIrDialect::LegacyLlvm7` | SM < 100 (Ampere, Ada, Hopper) | Typed: `i32*`, `float addrspace(1)*` | `NVPTX_DATALAYOUT_LEGACY` |
| `NvvmIrDialect::Modern` | SM >= 100 (Blackwell+) | Opaque: `ptr`, `ptr addrspace(1)` | `NVPTX_DATALAYOUT_FULL` |

The dialect is selected automatically from the target architecture via
`CudaArch::uses_legacy_llvm()` (threshold: `capability < 100`).

### What works today

The exporter already handles typed pointers in several structural positions
when `legacy_typed_pointers()` returns `true`:

1. **Function parameters and return types** -- `export_type()` in `types.rs`
   renders `PointerType` as `i8*` / `i8 addrspace(N)*` (the "canonical byte
   pointer") instead of `ptr`.

2. **Device-extern declarations** -- `DeviceExternType` in `externs.rs` carries
   full pointee information (e.g. `float*`, `[128 x i8] addrspace(1)*`) and
   `write_llvm()` renders the typed form for legacy mode.

3. **Memory instructions** -- `emit_load`, `emit_store`, `emit_alloca`,
   `emit_gep`, `emit_atomic_load/store/rmw/cmpxchg` all call
   `typed_pointer_operand()` which inserts a `bitcast i8* -> T*` before the
   instruction, then uses `export_pointer_to(pointee, addrspace)` for the
   pointer operand.

4. **Device-extern call sites** -- `device_extern_argument()` bitcasts internal
   `i8*` to the extern's declared pointer type, and the result pointer is
   bitcast back.

5. **GEP normalization** -- after a `getelementptr`, if the result pointee is
   not `i8`, a `bitcast T* -> i8*` normalizes back to the canonical form.

6. **Alloca normalization** -- `alloca T` produces `T*`; a bitcast back to
   `i8*` follows.

7. **AddressOf (globals and functions)** -- bitcasts from the global's natural
   type or the function's pointer type to `i8*`.

8. **@llvm.used** -- elements are `i8* bitcast(... @name to i8*)`.

9. **Metadata function references** -- `emit_function_reference` prints the
   full function-pointer type (`void (i8*, i8*)*`).

10. **Bitcast ptr->ptr** -- rewritten as `getelementptr i8, i8* %src, i64 0`
    because LLVM 7 has no ptr->ptr bitcast for opaque pointers.

11. **AddrSpaceCast, PtrToInt, IntToPtr** -- `export_cast()` delegates to
    `export_type()`, which prints the canonical `i8 addrspace(N)*`.

12. **Final validation** -- `verify_legacy_text()` scans the entire emitted
    string for bare `ptr` tokens and rejects the module if any are found.

### The canonical byte-pointer strategy

All internal pointers use **one canonical representation**: `i8*` (or
`i8 addrspace(N)*`).  When an instruction needs a differently-typed pointer
(e.g. `load float, float* %p`), the exporter inserts a `bitcast i8* -> float*`
immediately before use and (for instructions that produce pointers) a
`bitcast T* -> i8*` immediately after.

This strategy is sound because:
- The pliron IR is fully opaque-pointer: `PointerType` stores only an
  `address_space: u32`, with no pointee.
- Every load/store/GEP/alloca already carries a `TypeAttr` naming the
  accessed element type, so the pointee is available locally.
- Bitcasts between pointer types with the same address space are no-ops in
  NVVM IR.

### What is actually blocking

**Nothing is fundamentally blocking for standalone kernels (no device
externs).** The existing code already emits valid legacy LLVM 7 IR for pure
cuda-oxide kernels targeting SM_80/86/90. The `verify_legacy_text()` check
passes.

The limitation surfaces specifically when:

1. **External LTOIR linking is combined with legacy targets** -- the function
   bodies use `i8*` everywhere, which is legal LLVM 7 but may produce
   suboptimal code because libNVVM sees every pointer as `i8*` and must infer
   the true element type from usage.

2. **Potential type-mismatch at link boundaries** -- if the external LTOIR
   (compiled by nvcc) declares a function with `float addrspace(1)*` parameters
   and cuda-oxide's call site passes `i8 addrspace(1)*` (even with the
   adapter bitcasts), libNVVM's type checker may reject the module during LTO.
   The `device_extern_argument()` adapter avoids this for CALL instructions,
   but does not cover all possible cross-module interactions (e.g. function
   pointers stored in globals, callback patterns).

3. **Inline assembly constraints** -- PTX inline assembly that references
   pointer operands emits them with the canonical `i8*` type. If the assembly
   constraint expects a typed pointer to a specific element, the mismatch is
   silent at export time but may cause NVVM verification errors.

---

## 2. Catalog of Pointer-Emitting Sites

Every place in the export pipeline where a pointer type appears in the emitted
IR text:

### Already handled (typed in legacy mode)

| Site | File | How |
|------|------|-----|
| Function parameter types | `types.rs:29` | `export_type -> i8*` |
| Function return types | `types.rs:29` | `export_type -> i8*` |
| Load pointer operand | `ops.rs:635-644` | `typed_pointer_operand` bitcast + `export_pointer_to` |
| Store pointer operand | `ops.rs:672-683` | Same pattern |
| Alloca result | `ops.rs:721-746` | Alloca produces `T*`, bitcast to `i8*` |
| GEP base pointer | `ops.rs:867-906` | `typed_pointer_operand` + normalize result |
| Atomic load pointer | `ops.rs:928-937` | Same as load |
| Atomic store pointer | `ops.rs:960-971` | Same as store |
| AtomicRMW pointer | `ops.rs:997-1004` | Same pattern |
| AtomicCmpxchg pointer | `ops.rs:1034-1044` | Same pattern |
| Call (device extern) args | `ops.rs:1258-1285` | `device_extern_argument` bitcast |
| Call (device extern) result | `ops.rs:1289-1295` | `normalize_pointer_result` bitcast |
| Call (indirect) callee | `ops.rs:1236-1246` | Bitcast `i8*` to function-pointer type |
| Bitcast ptr->ptr | `ops.rs:1411-1420` | Rewritten as `gep i8, i8* %src, i64 0` |
| AddrSpaceCast | `ops.rs` | `export_cast` -> `export_type` -> canonical |
| PtrToInt | `ops.rs` | `export_cast` -> `export_type` -> canonical |
| IntToPtr | `ops.rs` | `export_cast` -> `export_type` -> canonical |
| AddressOf (global) | `ops.rs:1680-1696` | Bitcast `T addrspace(N)*` -> `i8 addrspace(N)*` |
| AddressOf (function) | `ops.rs:1703-1733` | Bitcast `ret(params)*` -> `i8*` |
| Global variables | `function.rs:56-118` | `export_type` for value type (not pointer, so not affected) |
| @llvm.used | `module.rs:240-255` | `i8* bitcast(... to i8*)` |
| Metadata annotations | `metadata.rs:121-125` | `export_function_pointer_type` |
| Select (pointer operands) | `ops.rs` | `export_type` -> canonical `i8*` |
| PHI nodes (pointer values) | `function.rs:654-680` | `export_type` -> canonical `i8*` |
| Struct fields with pointers | `types.rs:53-58` | Recursively calls `export_type` |
| Array of pointers | `types.rs:59-62` | Recursively calls `export_type` |

### Not handled (but currently not an issue)

| Site | Reason it works |
|------|-----------------|
| Inline asm operands | `export_type` prints `i8*`; constraint types are strings, not pointer types |
| Debug intrinsics (`llvm.dbg.declare`) | Legacy mode rejects debug metadata entirely |
| Vector of pointers | Not used in practice; `export_type` would print `<N x i8*>` correctly |

---

## 3. What libNVVM Actually Requires

### LLVM 7 dialect (CUDA 12.x, SM < 100)

- libNVVM's LLVM 7 parser **rejects `ptr`** as a keyword entirely. It does not
  exist in the LLVM 7 grammar. Any occurrence of a bare `ptr` token (outside
  strings, comments, or identifier names) causes a parse error.

- All pointers must be typed: `i32*`, `float addrspace(1)*`, etc.

- The specific pointee type matters only for type-checking across module
  boundaries (LTO). Within a single module, `i8*` is always valid and
  bitcasts are free. **libNVVM does not reject `i8*` as a pointer type.**

- A function declared as `declare float* @foo(float*)` must be called with
  `float*` arguments, not `i8*`. The adapter bitcast pattern that cuda-oxide
  uses is the correct solution.

### Modern dialect (CUDA 12.8+, SM >= 100)

- Accepts opaque `ptr` / `ptr addrspace(N)`.
- Typed pointers are still accepted for backward compatibility but deprecated.
- No pointee type is needed on load/store/GEP (the pointee is inferred from
  the instruction's type operand, which cuda-oxide always provides).

### Can we mix typed and opaque pointers?

**No.** LLVM (and libNVVM's parser) treats pointer syntax as a module-level
setting. A module cannot use `ptr` in one function and `i8*` in another.
The `opaque pointers` mode is all-or-nothing per compilation unit.

### Which CUDA toolkit version adds opaque pointer support?

- **CUDA 12.8** (libNVVM version shipped with CUDA 12.8) added opaque pointer
  support for SM_100+ (Blackwell). The NVVM IR spec version `2.0 / 3.2`
  corresponds to this.
- **Pre-12.8 toolkits** only accept typed pointers for all targets.
- There is **no** CUDA 12.x version that accepts opaque pointers for SM < 100.
  The LLVM 7 dialect is permanently typed-pointer-only for those targets.
- A future CUDA major version (13.x+) may deprecate the legacy dialect
  entirely, but as of CUDA 12.8 and the known CUDA 13 previews, SM < 100
  still requires LLVM 7 typed pointers through libNVVM.

---

## 4. Analysis: Is There Actually a Problem?

After thorough investigation, the current implementation is **functionally
correct** for the typed-pointer requirement. The `verify_legacy_text()` final
scan confirms no `ptr` tokens leak into the output. Here is what each concern
reduces to:

### 4a. Standalone kernels (no device externs)

**Status: Working.** The canonical `i8*` strategy produces valid LLVM 7 IR.
Load/store/GEP/alloca all emit properly typed pointer operands via the
`typed_pointer_operand()` adapter. The `verify_legacy_text()` check passes.

### 4b. Device externs with simple scalar/pointer parameters

**Status: Working.** The `device_extern_argument()` adapter bitcasts `i8*` to
the declared pointer type before the call, and `normalize_pointer_result`
bitcasts the result back. The declaration itself uses `DeviceExternType` with
full pointee information.

### 4c. Device externs with complex patterns (function pointers, callbacks)

**Status: Edge case, currently unsupported by the `DeviceExternType` schema.**
`DeviceExternType` does not have a `FunctionPointer` variant. If external
LTOIR ever needs a function-pointer parameter, it would need to be added.
This is a schema limitation, not a typed-pointer limitation.

### 4d. Code quality / optimization concern

libNVVM sees every internal pointer as `i8*` and must infer the true element
type from the bitcast/load/store chain. This is a normal pattern that LLVM's
type reconstruction handles without performance loss -- it is exactly what
Clang's `-mllvm -opaque-pointers=0` produces internally.

---

## 5. Proposed Approaches (If a Real Problem Surfaces)

### Option A: Propagate pointee types through IR values

**Approach:** Extend `PointerType` to optionally carry a pointee
`TypeHandle`, or maintain a side table
`FxHashMap<Value, TypeHandle /* pointee */>` during export.

**Files to change:**
- `crates/llvm-export/src/export/state.rs` -- add `pointer_pointees: FxHashMap<Value, TypeHandle>`
- `crates/llvm-export/src/export/types.rs` -- `export_type` reads pointee from the table
- `crates/llvm-export/src/export/ops.rs` -- every pointer-producing op populates the table
- `crates/llvm-export/src/export/function.rs` -- PHI nodes and block arguments need pointee propagation

**Tradeoffs:**
- (+) Most accurate: every pointer carries its true pointee type
- (-) Invasive: every pointer-producing op must be updated
- (-) Pointee ambiguity: phi nodes merging `float*` and `i32*` require an `i8*` fallback
- (-) The pliron IR has no pointee information, so the side table must be
  reconstructed from instruction semantics (load type, alloca element type, GEP
  source element type)
- (-) Unnecessary: the `i8*` + bitcast approach is already correct

**Estimated scope:** ~400-600 lines across 4 files; medium risk of regressions.

### Option B: Post-export text-level `ptr` -> typed-pointer rewriting

**Approach:** Parse the emitted LLVM IR text and replace `ptr` tokens with
inferred typed pointers based on adjacent instructions.

**Tradeoffs:**
- (+) Non-invasive: no changes to the structured export pipeline
- (-) Fragile: text parsing LLVM IR is error-prone
- (-) Unnecessary: the current code never emits `ptr` in legacy mode

**Estimated scope:** ~200-300 lines in a new module; high risk of edge cases.

### Option C: Rely on libNVVM's internal type reconstruction

**Approach:** Do nothing -- the current `i8*` + bitcast approach is what
Clang itself produces when targeting typed-pointer LLVM with opaque-pointer
source types.

**Tradeoffs:**
- (+) Zero implementation cost
- (+) Already working and validated by `verify_legacy_text()`
- (-) If libNVVM ever tightens validation, the `i8*` strategy may need revision

**Estimated scope:** Zero lines.

---

## 6. Recommendation

**Option C: No changes needed at this time.**

The investigation reveals that the "LTOIR typed-pointer limitation" is already
handled by the existing codebase. The canonical `i8*` + bitcast strategy
correctly satisfies libNVVM's LLVM 7 parser for all currently supported
patterns. The `verify_legacy_text()` final scan ensures no `ptr` tokens
escape.

### When this needs to be revisited

1. **If libNVVM rejects a specific `i8*` usage** -- add a targeted
   `typed_pointer_operand()` adapter for that instruction, following the
   existing pattern. This is a localized fix, not a redesign.

2. **If `DeviceExternType` needs new variants** (function pointers, opaque
   struct pointers) -- extend the enum and its `write_llvm()` method. The
   adapter bitcast pattern at call sites already handles this correctly.

3. **If CUDA drops the legacy dialect entirely** -- remove the legacy
   code paths. This simplifies the codebase significantly.

---

## 7. File Reference

Key files for any future typed-pointer work:

| File | Role |
|------|------|
| `crates/llvm-export/src/export/config.rs` | `NvvmIrDialect` enum, `NvvmExportConfig` |
| `crates/llvm-export/src/export/state.rs` | `ModuleExportState`, `legacy_typed_pointers()` flag |
| `crates/llvm-export/src/export/types.rs` | `export_type()` -- pointer rendering, `export_canonical_pointer_type()`, `export_pointer_to()` |
| `crates/llvm-export/src/export/externs.rs` | `DeviceExternType` with pointee info, `write_llvm()` |
| `crates/llvm-export/src/export/ops.rs` | `typed_pointer_operand()`, `device_extern_argument()`, all instruction emitters |
| `crates/llvm-export/src/export/function.rs` | Function/global export, alloca normalization |
| `crates/llvm-export/src/export/module.rs` | `verify_legacy_text()`, `contains_opaque_pointer_type()` |
| `crates/llvm-export/src/export/metadata.rs` | `emit_function_reference()` for typed function pointers |
| `crates/nvvm-transforms/src/legalize.rs` | Pre-export legalization for legacy NVVM |
| `crates/libnvvm-sys/src/lib.rs` | `CudaArch::uses_legacy_llvm()` dialect selection |
| `crates/mir-importer/src/pipeline.rs` | End-to-end pipeline, dialect resolution |
| `crates/llvm-export/tests/export_test.rs` | Legacy export integration tests |

---

## 8. Test Coverage

The existing test suite covers typed-pointer export comprehensively:

- `legacy_export_uses_one_canonical_pointer_with_multiple_typed_views` -- load/store with bitcasts
- `legacy_alloca_rejects_a_non_default_result_address_space` -- alloca validation
- `legacy_gep_rejects_a_result_address_space_different_from_its_base` -- GEP validation
- `legacy_pointer_select_keeps_one_canonical_type` -- select with pointer operands
- `legacy_device_extern_adapts_exact_pointer_arguments_and_results` -- call-site adaptation
- `legacy_device_extern_preserves_pointer_address_spaces` -- address-space fidelity
- `legacy_kernel_metadata_uses_typed_function_references` -- metadata emission
- `legacy_export_rejects_debug_metadata` -- unsupported feature gating
- `legacy_pointer_slot_is_recursively_canonical` -- nested struct/array pointer slots
- `legacy_function_address_defined_later_round_trips_through_indirect_call` -- function pointers
- `opaque_pointer_scan_ignores_names_strings_and_comments` -- `verify_legacy_text` correctness

All tests validate that zero `ptr` tokens appear in the final output.
