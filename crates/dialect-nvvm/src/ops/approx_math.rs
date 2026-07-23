/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Approximate math operations.
//!
//! Single-cycle hardware approximations that bypass libdevice:
//!
//! | Operation | PTX instruction | Min SM |
//! |---|---|---|
//! | [`TanhApproxF32Op`] | `tanh.approx.f32` | sm_75 |
//! | [`Ex2ApproxFtzF32Op`] | `ex2.approx.ftz.f32` | all |
//! | [`RcpApproxFtzF32Op`] | `rcp.approx.ftz.f32` | all |
//! | [`Lg2ApproxFtzF32Op`] | `lg2.approx.ftz.f32` | all |

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::FP32Type,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Verify that an operation has exactly one FP32 operand and one FP32 result.
fn verify_unary_f32(ctx: &Context, op: &Operation, name: &str) -> Result<(), Error> {
    if op.get_num_operands() != 1 || op.get_num_results() != 1 {
        return verify_err!(
            op.loc(),
            "{name} requires exactly one operand and one result"
        );
    }
    let f32_ty = FP32Type::get(ctx);
    let operand_ty = op.get_operand(0).get_type(ctx);
    if operand_ty != f32_ty.into() {
        return verify_err!(op.loc(), "{name} operand must be f32");
    }
    let result_ty = op.get_result(0).get_type(ctx);
    if result_ty != f32_ty.into() {
        return verify_err!(op.loc(), "{name} result must be f32");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Approximate hyperbolic tangent (`tanh.approx.f32`, sm_75+).
#[pliron_op(
    name = "nvvm.tanh_approx_f32",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct TanhApproxF32Op;

impl TanhApproxF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(ctx: &mut Context, input: Value) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![input],
            vec![],
            0,
        )
    }
}

impl Verify for TanhApproxF32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_unary_f32(
            ctx,
            &self.get_operation().deref(ctx),
            "nvvm.tanh_approx_f32",
        )
    }
}

/// Approximate base-2 exponential, flush-to-zero (`ex2.approx.ftz.f32`).
#[pliron_op(
    name = "nvvm.ex2_approx_ftz_f32",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct Ex2ApproxFtzF32Op;

impl Ex2ApproxFtzF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(ctx: &mut Context, input: Value) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![input],
            vec![],
            0,
        )
    }
}

impl Verify for Ex2ApproxFtzF32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_unary_f32(
            ctx,
            &self.get_operation().deref(ctx),
            "nvvm.ex2_approx_ftz_f32",
        )
    }
}

/// Approximate reciprocal, flush-to-zero (`rcp.approx.ftz.f32`).
#[pliron_op(
    name = "nvvm.rcp_approx_ftz_f32",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct RcpApproxFtzF32Op;

impl RcpApproxFtzF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(ctx: &mut Context, input: Value) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![input],
            vec![],
            0,
        )
    }
}

impl Verify for RcpApproxFtzF32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_unary_f32(
            ctx,
            &self.get_operation().deref(ctx),
            "nvvm.rcp_approx_ftz_f32",
        )
    }
}

/// Approximate base-2 logarithm, flush-to-zero (`lg2.approx.ftz.f32`).
#[pliron_op(
    name = "nvvm.lg2_approx_ftz_f32",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct Lg2ApproxFtzF32Op;

impl Lg2ApproxFtzF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(ctx: &mut Context, input: Value) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![input],
            vec![],
            0,
        )
    }
}

impl Verify for Lg2ApproxFtzF32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_unary_f32(
            ctx,
            &self.get_operation().deref(ctx),
            "nvvm.lg2_approx_ftz_f32",
        )
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub(super) fn register(ctx: &mut Context) {
    TanhApproxF32Op::register(ctx);
    Ex2ApproxFtzF32Op::register(ctx);
    RcpApproxFtzF32Op::register(ctx);
    Lg2ApproxFtzF32Op::register(ctx);
}
