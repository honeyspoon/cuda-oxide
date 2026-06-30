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

/// Warp MMA: m16n8k8, D=f32, A/B=bf16, C=f32 (smaller k variant).
///
/// Operands: `acc_ptr`, `a_ptr` (ptr to `[u32; 2]`), `b_ptr` (ptr to `u32`)
#[pliron_op(
    name = "nvvm.mma_m16n8k8_f32_bf16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K8F32Bf16Op;

impl MmaM16N8K8F32Bf16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K8F32Bf16Op { op }
    }
}

/// Warp MMA: m16n8k4, D=f32, A/B=tf32, C=f32 (smaller k variant).
///
/// Operands: `acc_ptr`, `a_ptr` (ptr to `[u32; 2]`), `b_ptr` (ptr to `u32`)
#[pliron_op(
    name = "nvvm.mma_m16n8k4_f32_tf32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K4F32Tf32Op;

impl MmaM16N8K4F32Tf32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K4F32Tf32Op { op }
    }
}

/// Warp MMA: m16n8k16, D=f32, A/B=f16, C=f32.
///
/// Operands: `acc_ptr`, `a_ptr` (ptr to `[u32; 4]`), `b_ptr` (ptr to `[u32; 2]`)
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f32_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K16F32F16Op;

impl MmaM16N8K16F32F16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F32F16Op { op }
    }
}

/// Warp MMA: m16n8k16, D=f16, A/B=f16, C=f16.
///
/// Operands: `acc_ptr`, `a_ptr` (ptr to `[u32; 4]`), `b_ptr` (ptr to `[u32; 2]`)
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K16F16Op;

impl MmaM16N8K16F16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F16Op { op }
    }
}

/// Warp MMA: m16n8k16, D=f16, A/B=f16, C=f32 (mixed accumulator).
///
/// Operands: `d_ptr`, `a_ptr` (ptr to `[u32; 4]`), `b_ptr` (ptr to `[u32; 2]`), `c_ptr`
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f16_f32acc",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>],
)]
pub struct MmaM16N8K16F16F32AccOp;

impl MmaM16N8K16F16F32AccOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F16F32AccOp { op }
    }
}

/// Warp MMA: m16n8k16, D=f32, A/B=f16, C=f16 (mixed accumulator).
///
/// Operands: `d_ptr`, `a_ptr` (ptr to `[u32; 4]`), `b_ptr` (ptr to `[u32; 2]`), `c_ptr`
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f32_f16acc",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>],
)]
pub struct MmaM16N8K16F32F16AccOp;

impl MmaM16N8K16F32F16AccOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F32F16AccOp { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MovmatrixTransB16Op::register(ctx);
    MmaM16N8K8F32Bf16Op::register(ctx);
    MmaM16N8K4F32Tf32Op::register(ctx);
    MmaM16N8K16F32F16Op::register(ctx);
    MmaM16N8K16F16Op::register(ctx);
    MmaM16N8K16F16F32AccOp::register(ctx);
    MmaM16N8K16F32F16AccOp::register(ctx);
}
