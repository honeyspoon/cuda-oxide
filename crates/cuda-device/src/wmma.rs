/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix load (ldmatrix) operations.
//!
//! These operations cooperatively load 8x8 matrix tiles from shared memory
//! into registers across a warp, matching the fragment layout expected by
//! tensor core `mma.sync` instructions.
//!
//! # Available Operations
//!
//! ```text
//! ┌─────────────────────┬───────┬──────────┬───────────┬─────────────────────────┐
//! │ Operation           │ Tiles │ Regs/Thr │ Transpose │ PTX                     │
//! ├─────────────────────┼───────┼──────────┼───────────┼─────────────────────────┤
//! │ ldmatrix_x1         │ 1     │ 1        │ No        │ ldmatrix...m8n8.x1      │
//! │ ldmatrix_x1_trans   │ 1     │ 1        │ Yes       │ ldmatrix...x1.trans     │
//! └─────────────────────┴───────┴──────────┴───────────┴─────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - **Execution**: Warp-synchronous (all 32 threads must participate)
//! - **Memory**: Source must be in shared memory
//! - **Alignment**: Pointer must be 16-byte aligned

/// Load one 8x8 matrix tile from shared memory.
///
/// Warp-cooperative matrix load that loads a single 8x8 tile from shared
/// memory into one register per thread. Each of the 32 warp threads
/// provides its own shared memory address, and the hardware cooperatively
/// loads the full 8x8 tile.
///
/// # PTX Instruction
///
/// `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r0}, [addr];`
///
/// # Arguments
///
/// - `smem_ptr`: Source pointer in shared memory (16-byte aligned)
///
/// # Returns
///
/// A single `u32` register containing 2 packed b16 values from the matrix.
///
/// # Safety
///
/// - `smem_ptr` must be a valid, 16-byte-aligned shared memory pointer.
/// - Must be called by all 32 threads in a warp together.
///
/// See also: [`ldmatrix_x1_trans`]
#[inline(never)]
pub unsafe fn ldmatrix_x1(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1 called outside CUDA kernel context")
}

/// Load one 8x8 matrix tile from shared memory with transpose.
///
/// Warp-cooperative matrix load with the `.trans` modifier that transposes
/// the data during the load. This converts from row-major shared memory
/// layout to the column-major fragment layout expected by certain
/// `mma.sync` operands.
///
/// # PTX Instruction
///
/// `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r0}, [addr];`
///
/// # Arguments
///
/// - `smem_ptr`: Source pointer in shared memory (16-byte aligned)
///
/// # Returns
///
/// A single `u32` register containing 2 packed b16 values from the
/// transposed matrix.
///
/// # Safety
///
/// - `smem_ptr` must be a valid, 16-byte-aligned shared memory pointer.
/// - Must be called by all 32 threads in a warp together.
///
/// See also: [`ldmatrix_x1`]
#[inline(never)]
pub unsafe fn ldmatrix_x1_trans(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1_trans called outside CUDA kernel context")
}
