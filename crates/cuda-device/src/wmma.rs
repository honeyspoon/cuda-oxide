/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp Matrix (WMMA) store intrinsics.
//!
//! These intrinsics provide warp-cooperative matrix stores to shared memory
//! using the `stmatrix` PTX instruction family. They properly handle tensor
//! core fragment layouts when writing results to shared memory.
//!
//! # Variants
//!
//! | Function             | Tiles | Transpose | PTX                                              |
//! |----------------------|-------|-----------|--------------------------------------------------|
//! | `stmatrix_x4`        | 4     | No        | `stmatrix.sync.aligned.m8n8.x4.shared.b16`       |
//! | `stmatrix_x2`        | 2     | No        | `stmatrix.sync.aligned.m8n8.x2.shared.b16`       |
//! | `stmatrix_x4_trans`  | 4     | Yes       | `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16`  |
//! | `stmatrix_x2_trans`  | 2     | Yes       | `stmatrix.sync.aligned.m8n8.x2.trans.shared.b16`  |
//!
//! # Safety
//!
//! All functions require:
//! - `smem_ptr` must point to valid shared memory, aligned to the tile size
//! - Must be called by all 32 threads in a warp (warp-synchronous)
//! - `data` elements must contain properly packed 16-bit values (e.g. bf16x2)

/// Store four 8x8 matrix tiles to shared memory.
///
/// Performs a warp-cooperative store of 4 matrix tiles (256 elements total)
/// to shared memory without transposition.
///
/// # Parameters
///
/// - `smem_ptr`: Destination pointer in shared memory (16-byte aligned)
/// - `data`: Array of 4 register values, each containing 2 packed 16-bit elements
///
/// # PTX Instruction
///
/// ```ptx
/// stmatrix.sync.aligned.m8n8.x4.shared.b16 [addr], {r0, r1, r2, r3};
/// ```
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together
#[inline(never)]
pub unsafe fn stmatrix_x4(smem_ptr: *mut u32, data: &[u32; 4]) {
    let _ = (smem_ptr, data);
    unreachable!("stmatrix_x4 called outside CUDA kernel context")
}

/// Store two 8x8 matrix tiles to shared memory.
///
/// Performs a warp-cooperative store of 2 matrix tiles (128 elements total)
/// to shared memory without transposition.
///
/// # Parameters
///
/// - `smem_ptr`: Destination pointer in shared memory (16-byte aligned)
/// - `data`: Array of 2 register values, each containing 2 packed 16-bit elements
///
/// # PTX Instruction
///
/// ```ptx
/// stmatrix.sync.aligned.m8n8.x2.shared.b16 [addr], {r0, r1};
/// ```
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together
#[inline(never)]
pub unsafe fn stmatrix_x2(smem_ptr: *mut u32, data: &[u32; 2]) {
    let _ = (smem_ptr, data);
    unreachable!("stmatrix_x2 called outside CUDA kernel context")
}

/// Store four 8x8 matrix tiles to shared memory with transpose.
///
/// Performs a warp-cooperative store of 4 matrix tiles (256 elements total)
/// to shared memory with the `.trans` modifier that transforms data from
/// fragment layout to row-major layout.
///
/// # Parameters
///
/// - `smem_ptr`: Destination pointer in shared memory (16-byte aligned)
/// - `data`: Array of 4 register values, each containing 2 packed 16-bit elements
///
/// # PTX Instruction
///
/// ```ptx
/// stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [addr], {r0, r1, r2, r3};
/// ```
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together
/// - Registers must contain properly packed bf16 pairs
#[inline(never)]
pub unsafe fn stmatrix_x4_trans(smem_ptr: *mut u32, data: &[u32; 4]) {
    let _ = (smem_ptr, data);
    unreachable!("stmatrix_x4_trans called outside CUDA kernel context")
}

/// Store two 8x8 matrix tiles to shared memory with transpose.
///
/// Performs a warp-cooperative store of 2 matrix tiles (128 elements total)
/// to shared memory with the `.trans` modifier that transforms data from
/// fragment layout to row-major layout.
///
/// # Parameters
///
/// - `smem_ptr`: Destination pointer in shared memory (16-byte aligned)
/// - `data`: Array of 2 register values, each containing 2 packed 16-bit elements
///
/// # PTX Instruction
///
/// ```ptx
/// stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [addr], {r0, r1};
/// ```
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together
/// - Registers must contain properly packed bf16 pairs
#[inline(never)]
pub unsafe fn stmatrix_x2_trans(smem_ptr: *mut u32, data: &[u32; 2]) {
    let _ = (smem_ptr, data);
    unreachable!("stmatrix_x2_trans called outside CUDA kernel context")
}
