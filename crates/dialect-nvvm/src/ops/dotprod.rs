// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer dot product operations (`dp4a`, `dp2a`).
//!
//! These are single-thread, non-convergent packed integer dot product
//! instructions lowered to inline PTX. Available from `sm_61+`.

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

/// Shared verifier for dot product ops: checks 3 operands + 1 result, all i32.
fn verify_dp_op(ctx: &Context, op: &Operation, op_name: &str) -> Result<(), Error> {
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != 3 || op.get_num_results() != 1 {
        return verify_err!(
            op.loc(),
            "{} requires exactly 3 operands and 1 result",
            op_name
        );
    }

    for (i, operand) in operands.iter().enumerate() {
        let ty = operand.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        let Some(integer) = ty_ref.downcast_ref::<IntegerType>() else {
            return verify_err!(
                op.loc(),
                "{} operand {} must be a 32-bit integer",
                op_name,
                i
            );
        };
        if integer.width() != 32 {
            return verify_err!(
                op.loc(),
                "{} operand {} must be a 32-bit integer",
                op_name,
                i
            );
        }
    }

    let res_ty = op.get_result(0).get_type(ctx);
    let res_ref = res_ty.deref(ctx);
    let Some(integer) = res_ref.downcast_ref::<IntegerType>() else {
        return verify_err!(op.loc(), "{} result must be a 32-bit integer", op_name);
    };
    if integer.width() != 32 {
        return verify_err!(op.loc(), "{} result must be a 32-bit integer", op_name);
    }

    Ok(())
}

/// Signed 4-element byte dot product with accumulation: `d = c + dot(a, b)`.
///
/// `a` and `b` are each 4 packed signed bytes; `c` and `d` are signed 32-bit.
///
/// PTX: `dp4a.s32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 4×i8
/// - `b` (u32): packed 4×i8
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp4a_s32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aS32Op;

impl Dp4aS32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aS32Op { op }
    }
}

/// Unsigned 4-element byte dot product with accumulation: `d = c + dot(a, b)`.
///
/// `a` and `b` are each 4 packed unsigned bytes; `c` and `d` are unsigned 32-bit.
///
/// PTX: `dp4a.u32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 4×u8
/// - `b` (u32): packed 4×u8
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp4a_u32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aU32Op;

impl Dp4aU32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aU32Op { op }
    }
}

/// Signed 2-element half-word × byte dot product (lower half): `d = c + dot(a, b)`.
///
/// `a` is 2 packed signed 16-bit values; `b`'s lower 2 bytes are used.
///
/// PTX: `dp2a.lo.s32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 2×i16
/// - `b` (u32): packed bytes (lower 2 used)
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_s32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aS32Op;

impl Dp2aS32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aS32Op { op }
    }
}

/// Unsigned 2-element half-word × byte dot product (lower half): `d = c + dot(a, b)`.
///
/// `a` is 2 packed unsigned 16-bit values; `b`'s lower 2 bytes are used.
///
/// PTX: `dp2a.lo.u32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 2×u16
/// - `b` (u32): packed bytes (lower 2 used)
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_u32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aU32Op;

impl Dp2aU32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aU32Op { op }
    }
}

/// Signed 2-element half-word × byte dot product (upper half): `d = c + dot(a, b)`.
///
/// `a` is 2 packed signed 16-bit values; `b`'s upper 2 bytes are used.
///
/// PTX: `dp2a.hi.s32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (i32): packed 2×i16
/// - `b` (u32): packed bytes (upper 2 used)
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_hi_s32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aHiS32Op;

impl Verify for Dp2aHiS32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_dp_op(ctx, &self.get_operation().deref(ctx), "nvvm.dp2a_hi_s32")
    }
}

impl Dp2aHiS32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aHiS32Op { op }
    }
}

/// Unsigned 2-element half-word × byte dot product (upper half): `d = c + dot(a, b)`.
///
/// `a` is 2 packed unsigned 16-bit values; `b`'s upper 2 bytes are used.
///
/// PTX: `dp2a.hi.u32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 2×u16
/// - `b` (u32): packed bytes (upper 2 used)
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_hi_u32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aHiU32Op;

impl Verify for Dp2aHiU32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_dp_op(ctx, &self.get_operation().deref(ctx), "nvvm.dp2a_hi_u32")
    }
}

impl Dp2aHiU32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aHiU32Op { op }
    }
}

/// Mixed-signedness 4-element byte dot product with accumulation.
///
/// `a` is 4 packed signed bytes; `b` is 4 packed unsigned bytes.
///
/// PTX: `dp4a.s32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (i32): packed 4×i8
/// - `b` (u32): packed 4×u8
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp4a_s32_u32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aS32U32Op;

impl Verify for Dp4aS32U32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_dp_op(ctx, &self.get_operation().deref(ctx), "nvvm.dp4a_s32_u32")
    }
}

impl Dp4aS32U32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aS32U32Op { op }
    }
}

/// Mixed-signedness 4-element byte dot product with accumulation.
///
/// `a` is 4 packed unsigned bytes; `b` is 4 packed signed bytes.
///
/// PTX: `dp4a.u32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 4×u8
/// - `b` (i32): packed 4×i8
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp4a_u32_s32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aU32S32Op;

impl Verify for Dp4aU32S32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_dp_op(ctx, &self.get_operation().deref(ctx), "nvvm.dp4a_u32_s32")
    }
}

impl Dp4aU32S32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aU32S32Op { op }
    }
}

/// Mixed-signedness 2-element half-word × byte dot product (lower half).
///
/// `a` is 2 packed signed 16-bit values; `b`'s lower 2 bytes are unsigned.
///
/// PTX: `dp2a.lo.s32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (i32): packed 2×i16
/// - `b` (u32): packed bytes (lower 2 used, unsigned)
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_lo_s32_u32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aLoS32U32Op;

impl Verify for Dp2aLoS32U32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_dp_op(
            ctx,
            &self.get_operation().deref(ctx),
            "nvvm.dp2a_lo_s32_u32",
        )
    }
}

impl Dp2aLoS32U32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aLoS32U32Op { op }
    }
}

/// Mixed-signedness 2-element half-word × byte dot product (lower half).
///
/// `a` is 2 packed unsigned 16-bit values; `b`'s lower 2 bytes are signed.
///
/// PTX: `dp2a.lo.u32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 2×u16
/// - `b` (i32): packed bytes (lower 2 used, signed)
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_lo_u32_s32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aLoU32S32Op;

impl Verify for Dp2aLoU32S32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_dp_op(
            ctx,
            &self.get_operation().deref(ctx),
            "nvvm.dp2a_lo_u32_s32",
        )
    }
}

impl Dp2aLoU32S32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aLoU32S32Op { op }
    }
}

/// Register dot product operations with the context.
pub(super) fn register(ctx: &mut Context) {
    Dp4aS32Op::register(ctx);
    Dp4aU32Op::register(ctx);
    Dp2aS32Op::register(ctx);
    Dp2aU32Op::register(ctx);
    Dp2aHiS32Op::register(ctx);
    Dp2aHiU32Op::register(ctx);
    Dp4aS32U32Op::register(ctx);
    Dp4aU32S32Op::register(ctx);
    Dp2aLoS32U32Op::register(ctx);
    Dp2aLoU32S32Op::register(ctx);
}
