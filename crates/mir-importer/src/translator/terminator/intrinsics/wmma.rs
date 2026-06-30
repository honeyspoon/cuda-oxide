/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsics (`movmatrix`).

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::MovmatrixTransB16Op;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit movmatrix_trans_b16: in-register 8×8 matrix transpose.
///
/// Takes one u32 operand and returns one u32.
#[allow(clippy::too_many_arguments)]
pub fn emit_movmatrix_trans_b16(
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
                "movmatrix_trans_b16 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (a_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let mov_op = Operation::new(
        ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val],
        vec![],
        0,
    );
    mov_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mov_op.insert_after(ctx, prev);
    } else {
        mov_op.insert_at_front(block_ptr, ctx);
    }

    let result = mov_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        mov_op,
        value_map,
        block_map,
        loc,
        "movmatrix_trans_b16 call without target block",
    )
}

/// Emit `mma_m16n8k8_f32_bf16`: bf16 m16n8k8 (smaller k).
pub fn emit_mma_m16n8k8_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_3op!(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        MmaM16N8K8F32Bf16Op,
        "mma_m16n8k8_f32_bf16"
    )
}

/// Emit `mma_m16n8k4_f32_tf32`: tf32 m16n8k4 (smaller k).
pub fn emit_mma_m16n8k4_f32_tf32(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_3op!(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        MmaM16N8K4F32Tf32Op,
        "mma_m16n8k4_f32_tf32"
    )
}

/// Emit `mma_m16n8k16_f32_f16`: f16 inputs, f32 accumulator.
pub fn emit_mma_m16n8k16_f32_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_3op!(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        MmaM16N8K16F32F16Op,
        "mma_m16n8k16_f32_f16"
    )
}

/// Emit `mma_m16n8k16_f16`: f16 inputs, f16 accumulator.
pub fn emit_mma_m16n8k16_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_3op!(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        MmaM16N8K16F16Op,
        "mma_m16n8k16_f16"
    )
}

/// Emit `mma_m16n8k16_f16_f32acc`: D=f16, C=f32 (4 operands).
pub fn emit_mma_m16n8k16_f16_f32acc(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_4op!(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        MmaM16N8K16F16F32AccOp,
        "mma_m16n8k16_f16_f32acc"
    )
}

/// Emit `mma_m16n8k16_f32_f16acc`: D=f32, C=f16 (4 operands).
pub fn emit_mma_m16n8k16_f32_f16acc(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_mma_4op!(
        ctx,
        body,
        args,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        MmaM16N8K16F32F16AccOp,
        "mma_m16n8k16_f32_f16acc"
    )
}