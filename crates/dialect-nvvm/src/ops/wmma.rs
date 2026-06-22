/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared memory matrix load (ldmatrix) operations.
//!
//! Ldmatrix provides warp-cooperative matrix load operations that properly
//! handle tensor core fragment layouts when reading from shared memory.
//!
//! ```text
//! ┌─────────────────────┬───────┬──────────┬───────────┬──────────────────────────┐
//! │ Operation           │ Tiles │ Regs/Thr │ Transpose │ PTX                      │
//! ├─────────────────────┼───────┼──────────┼───────────┼──────────────────────────┤
//! │ LdmatrixX1Op        │ 1     │ 1        │ No        │ ldmatrix...m8n8.x1       │
//! │ LdmatrixX1TransOp   │ 1     │ 1        │ Yes       │ ldmatrix...x1.trans      │
//! └─────────────────────┴───────┴──────────┴───────────┴──────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - **Execution**: Warp-synchronous (all 32 threads must participate)
//! - **Memory**: Source must be in shared memory
//! - **Alignment**: Pointer must be aligned to tile size

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Load one 8x8 matrix tile from shared memory.
///
/// Warp-cooperative matrix load without transpose.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r0}, [addr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
///
/// # Results
///
/// - `r0` (i32): loaded register value (2 packed b16 values)
#[pliron_op(
    name = "nvvm.ldmatrix_x1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LdmatrixX1Op;

impl LdmatrixX1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX1Op { op }
    }
}

/// Load one 8x8 matrix tile from shared memory with transpose.
///
/// Warp-cooperative matrix load with the `.trans` modifier that transforms
/// data during load from row-major to column-major fragment layout.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r0}, [addr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
///
/// # Results
///
/// - `r0` (i32): loaded register value (2 packed b16 values)
#[pliron_op(
    name = "nvvm.ldmatrix_x1_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LdmatrixX1TransOp;

impl LdmatrixX1TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX1TransOp { op }
    }
}

/// Register ldmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    LdmatrixX1Op::register(ctx);
    LdmatrixX1TransOp::register(ctx);
}
