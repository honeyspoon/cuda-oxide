/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bit-manipulation PTX operations.
//!
//! | Operation  | PTX Instruction | Description                      |
//! |------------|-----------------|----------------------------------|
//! | `PrmtB32`  | `prmt.b32`      | Byte permute on two 32-bit words |
//!
//! # Requirements
//!
//! - **PTX ISA**: 2.0+
//! - **Architecture**: sm_20+ (all modern GPUs)

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

/// Byte permute: rearrange bytes from two 32-bit words.
///
/// Selects four bytes from the concatenation of `a` (high) and `b` (low)
/// according to the control word `c`, producing a new 32-bit value.
///
/// This is a pure per-thread operation (not convergent).
///
/// PTX: `prmt.b32 $0, $1, $2, $3;`
///
/// # Operands
///
/// - `a` (i32): upper source word
/// - `b` (i32): lower source word
/// - `c` (i32): control word (byte selectors)
///
/// # Results
///
/// - `result` (i32): permuted output
///
/// # Verification
///
/// - Must have 3 operands and 1 result of type `i32`
#[pliron_op(
    name = "nvvm.prmt_b32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct PrmtB32Op;

impl PrmtB32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        PrmtB32Op { op }
    }
}

impl Verify for PrmtB32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.prmt_b32 result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(op.loc(), "nvvm.prmt_b32 result must be 32-bit integer");
        }
        Ok(())
    }
}

/// Register bitops operations with the context.
pub(super) fn register(ctx: &mut Context) {
    PrmtB32Op::register(ctx);
}
