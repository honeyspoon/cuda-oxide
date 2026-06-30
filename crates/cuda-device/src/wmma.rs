/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operations.
//!
//! This module provides a register-only 8×8 transpose (`movmatrix`) and
//! warp-cooperative shared-memory loads (`ldmatrix`).
//!
//! For ldmatrix, each group of four lanes loads one naturally aligned
//! 16-byte row:
//!
//! ```text
//! x1: lanes  0..7  provide addresses
//! x2: lanes  0..15 provide addresses
//! x4: lanes  0..31 provide addresses
//! ```
//!
//! On sm_75, x1 and x2 still require valid addresses in all 32 lanes. A
//! common choice is to copy the lower-lane addresses into the upper lanes.
//! The .trans forms use column-major rather than row-major layout.
//!
//! Ldmatrix is a weak memory operation: .sync converges the warp but does not
//! order memory. Callers need an appropriate barrier or fence around dependent
//! memory accesses. Movmatrix is register-only and has no memory effect.

/// Transpose an 8×8 matrix of b16 elements in-register across the warp.
///
/// Each lane provides one `u32` that packs two b16 elements of the source
/// matrix. The instruction collectively transposes the 8×8 tile and writes
/// the transposed pair back into each lane's destination register.
///
/// ```text
/// input  lane 4*r + k: [matrix[r][2*k], matrix[r][2*k + 1]]
/// output lane 4*c + k: [matrix[2*k][c], matrix[2*k + 1][c]]
/// ```
///
/// This operation only exchanges register fragments between lanes. It does
/// not access memory and is not a memory fence.
///
/// # PTX
///
/// `movmatrix.sync.aligned.m8n8.trans.b16 %d, %a;`
///
/// # Safety
///
/// - All 32 lanes must execute the same call together.
/// - Calling from divergent control flow is undefined behavior.
/// - Requires `sm_75+` and PTX ISA 7.8+. cuda-oxide selects both floors
///   automatically, including when targeting Turing or Ampere.
#[inline(never)]
#[must_use]
pub unsafe fn movmatrix_trans_b16(a: u32) -> u32 {
    let _ = a;
    unreachable!("movmatrix_trans_b16 called outside CUDA kernel context")
}

// =============================================================================
// Shared-memory matrix loads
// =============================================================================

/// Load one 8×8 matrix tile from shared memory.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r0}, [addr];`
///
/// # Safety
///
/// - Lanes 0-7 must each provide a valid, naturally aligned 16-byte shared-memory row
/// - On sm_75, all 32 lanes must provide a valid address
/// - Must be called by all threads in a warp (warp-synchronous)
/// - Callers must use a suitable barrier or fence to order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x1(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1 called outside CUDA kernel context")
}

/// Load one 8×8 matrix tile from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r0}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x1`].
#[inline(never)]
pub unsafe fn ldmatrix_x1_trans(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1_trans called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory.
///
/// Returns `[u32; 2]` (each u32 = 2 packed b16 values).
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0, %r1}, [addr];`
///
/// # Safety
///
/// - Lanes 0-15 provide the 16 row addresses
/// - On sm_75, all 32 lanes must provide a valid address
/// - All lanes must participate, and callers must order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x2(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2 called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r0, %r1}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x2`].
#[inline(never)]
pub unsafe fn ldmatrix_x2_trans(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2_trans called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory.
///
/// Returns `[u32; 4]` (each u32 = 2 packed b16 values).
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Safety
///
/// - All 32 lanes provide valid, naturally aligned 16-byte row addresses
/// - All lanes must participate, and callers must order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x4(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4 called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x4`].
#[inline(never)]
pub unsafe fn ldmatrix_x4_trans(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4_trans called outside CUDA kernel context")
}

/// Warp MMA: D = A x B + C (m16n8k8, f32 output, bf16 inputs).
///
/// Performs a 16x8x8 matrix multiplication using tensor cores with bf16 input
/// fragments and f32 accumulator. Smaller k variant of m16n8k16.
///
/// # Matrix Dimensions
///
/// - **A**: 16x8 (row-major, bf16), distributed as 2 x u32 per thread
/// - **B**: 8x8 (col-major, bf16), distributed as 1 x u32 per thread
/// - **D/C**: 16x8 (f32 accumulator), distributed as 4 x f32 per thread
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (4 x f32, read-modify-write)
/// - `a`: A fragment (2 x u32, each u32 = 2 packed bf16)
/// - `b`: B fragment (1 x u32, containing 2 packed bf16)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1},
///     {%b0},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k8_f32_bf16(acc: &mut [f32; 4], a: &[u32; 2], b: &u32) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k8_f32_bf16 called outside CUDA kernel context")
}

/// Warp MMA: D = A x B + C (m16n8k4, f32 output, tf32 inputs).
///
/// Performs a 16x8x4 matrix multiplication using tensor cores with tf32 input
/// fragments and f32 accumulator. Smaller k variant of m16n8k8.
///
/// # Matrix Dimensions
///
/// - **A**: 16x4 (row-major, tf32), distributed as 2 x u32 per thread
/// - **B**: 4x8 (col-major, tf32), distributed as 1 x u32 per thread
/// - **D/C**: 16x8 (f32 accumulator), distributed as 4 x f32 per thread
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (4 x f32, read-modify-write)
/// - `a`: A fragment (2 x u32, each u32 = 1 tf32 value)
/// - `b`: B fragment (1 x u32, containing 1 tf32 value)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1},
///     {%b0},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k4_f32_tf32(acc: &mut [f32; 4], a: &[u32; 2], b: &u32) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k4_f32_tf32 called outside CUDA kernel context")
}

/// Warp MMA: D = A x B + C (m16n8k16, f32 output, f16 inputs).
///
/// Performs a 16x8x16 matrix multiplication using tensor cores with f16 input
/// fragments and f32 accumulator.
///
/// # Matrix Dimensions
///
/// - **A**: 16x16 (row-major, f16), distributed as 4 x u32 per thread
/// - **B**: 16x8 (col-major, f16), distributed as 2 x u32 per thread
/// - **D/C**: 16x8 (f32 accumulator), distributed as 4 x f32 per thread
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (4 x f32, read-modify-write)
/// - `a`: A fragment (4 x u32, each u32 = 2 packed f16)
/// - `b`: B fragment (2 x u32, each u32 = 2 packed f16)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k16_f32_f16(acc: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k16_f32_f16 called outside CUDA kernel context")
}

/// Warp MMA: D = A x B + C (m16n8k16, f16 output, f16 inputs).
///
/// Performs a 16x8x16 matrix multiplication using tensor cores with f16 input
/// fragments and f16 accumulator.
///
/// # Matrix Dimensions
///
/// - **A**: 16x16 (row-major, f16), distributed as 4 x u32 per thread
/// - **B**: 16x8 (col-major, f16), distributed as 2 x u32 per thread
/// - **D/C**: 16x8 (f16 accumulator), distributed as 2 x u32 per thread (packed f16)
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (2 x u32, read-modify-write, each u32 = 2 packed f16)
/// - `a`: A fragment (4 x u32, each u32 = 2 packed f16)
/// - `b`: B fragment (2 x u32, each u32 = 2 packed f16)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
///     {%d0, %d1},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k16_f16(acc: &mut [u32; 2], a: &[u32; 4], b: &[u32; 2]) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k16_f16 called outside CUDA kernel context")
}

/// Warp MMA: D(f16) = A x B + C(f32) (m16n8k16, mixed accumulator).
///
/// Performs a 16x8x16 matrix multiplication using tensor cores with f16 input
/// fragments, f32 source accumulator (C), and f16 destination (D).
///
/// # Matrix Dimensions
///
/// - **A**: 16x16 (row-major, f16), distributed as 4 x u32 per thread
/// - **B**: 16x8 (col-major, f16), distributed as 2 x u32 per thread
/// - **C**: 16x8 (f32 accumulator), distributed as 4 x f32 per thread
/// - **D**: 16x8 (f16 result), distributed as 2 x u32 per thread (packed f16)
///
/// # Parameters
///
/// - `d`: Destination (2 x u32, packed f16 output)
/// - `a`: A fragment (4 x u32, each u32 = 2 packed f16)
/// - `b`: B fragment (2 x u32, each u32 = 2 packed f16)
/// - `c`: Accumulator (4 x f32, read-only)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f32
///     {%d0, %d1},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k16_f16_f32acc(d: &mut [u32; 2], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    let _ = (d, a, b, c);
    unreachable!("mma_m16n8k16_f16_f32acc called outside CUDA kernel context")
}

/// Warp MMA: D(f32) = A x B + C(f16) (m16n8k16, mixed accumulator).
///
/// Performs a 16x8x16 matrix multiplication using tensor cores with f16 input
/// fragments, f16 source accumulator (C), and f32 destination (D).
///
/// # Matrix Dimensions
///
/// - **A**: 16x16 (row-major, f16), distributed as 4 x u32 per thread
/// - **B**: 16x8 (col-major, f16), distributed as 2 x u32 per thread
/// - **C**: 16x8 (f16 accumulator), distributed as 2 x u32 per thread (packed f16)
/// - **D**: 16x8 (f32 result), distributed as 4 x f32 per thread
///
/// # Parameters
///
/// - `d`: Destination (4 x f32 output)
/// - `a`: A fragment (4 x u32, each u32 = 2 packed f16)
/// - `b`: B fragment (2 x u32, each u32 = 2 packed f16)
/// - `c`: Accumulator (2 x u32, read-only, each u32 = 2 packed f16)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f16
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k16_f32_f16acc(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[u32; 2]) {
    let _ = (d, a, b, c);
    unreachable!("mma_m16n8k16_f32_f16acc called outside CUDA kernel context")
}