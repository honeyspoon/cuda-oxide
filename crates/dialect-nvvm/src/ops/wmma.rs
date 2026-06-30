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

/// Warp MMA: m16n8k128 with s32 accumulator and b1 inputs (xor.popc).
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to `[i32; 4]` accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to `[u32; 2]` A fragment (packed b1)
/// - `b_ptr` (ptr): pointer to `u32` B fragment (packed b1)
///
/// # Results
///
/// - None (accumulator updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m16n8k128_s32_b1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K128S32B1Op;

impl MmaM16N8K128S32B1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K128S32B1Op { op }
    }
}

/// Warp MMA: m16n8k256 with s32 accumulator and b1 inputs (xor.popc).
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to `[i32; 4]` accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to `[u32; 4]` A fragment (packed b1)
/// - `b_ptr` (ptr): pointer to `[u32; 2]` B fragment (packed b1)
///
/// # Results
///
/// - None (accumulator updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m16n8k256_s32_b1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K256S32B1Op;

impl MmaM16N8K256S32B1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K256S32B1Op { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MovmatrixTransB16Op::register(ctx);
    MmaM16N8K128S32B1Op::register(ctx);
    MmaM16N8K256S32B1Op::register(ctx);
}
