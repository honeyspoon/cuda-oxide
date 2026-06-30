/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix dialect operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

/// In-register 8×8 matrix transpose (movmatrix.sync.aligned.m8n8.trans.b16).
#[pliron_op(
    name = "nvvm.movmatrix_trans_b16",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct MovmatrixTransB16Op;

impl MovmatrixTransB16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for MovmatrixTransB16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);

        if op.operands().count() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.movmatrix_trans_b16 requires one operand and one result"
            );
        }

        for (name, ty) in [
            ("operand", op.get_operand(0).get_type(ctx)),
            ("result", op.get_result(0).get_type(ctx)),
        ] {
            let ty_ref = ty.deref(ctx);
            let Some(integer) = ty_ref.downcast_ref::<IntegerType>() else {
                return verify_err!(
                    op.loc(),
                    "nvvm.movmatrix_trans_b16 {} must be a 32-bit integer",
                    name
                );
            };
            if integer.width() != 32 {
                return verify_err!(
                    op.loc(),
                    "nvvm.movmatrix_trans_b16 {} must be a 32-bit integer",
                    name
                );
            }
        }

        Ok(())
    }
}

// =============================================================================
// Signed int4 operations
// =============================================================================

/// MMA m16n8k32 with signed int4 (s4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 2×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 1×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k32_s32_s4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K32S32S4Op;

impl MmaM16N8K32S32S4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K32S32S4Op { op }
    }
}

/// MMA m16n8k64 with signed int4 (s4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 4×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 2×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k64_s32_s4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K64S32S4Op;

impl MmaM16N8K64S32S4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K64S32S4Op { op }
    }
}

// =============================================================================
// Unsigned int4 operations
// =============================================================================

/// MMA m16n8k32 with unsigned int4 (u4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 2×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 1×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k32_s32_u4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K32S32U4Op;

impl MmaM16N8K32S32U4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K32S32U4Op { op }
    }
}

/// MMA m16n8k64 with unsigned int4 (u4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 4×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 2×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k64_s32_u4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K64S32U4Op;

impl MmaM16N8K64S32U4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K64S32U4Op { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MovmatrixTransB16Op::register(ctx);
    MmaM16N8K32S32S4Op::register(ctx);
    MmaM16N8K32S32U4Op::register(ctx);
    MmaM16N8K64S32S4Op::register(ctx);
    MmaM16N8K64S32U4Op::register(ctx);
}
