/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Translation helper for approximate math intrinsics.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit a unary approximate-math operation (1 f32 operand → 1 f32 result).
pub fn emit_approx_math_unary<O: Op>(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "approximate math intrinsic expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (input, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let f32_ty = FP32Type::get(ctx);
    let op = Operation::new(
        ctx,
        O::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![input],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "approximate math intrinsic call without target block",
    )
}
