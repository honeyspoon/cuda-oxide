/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA / ldmatrix intrinsics.
//!
//! Translates `cuda_device::wmma::ldmatrix_*` intrinsic calls into
//! `dialect-nvvm` ldmatrix operations.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{LdmatrixX1Op, LdmatrixX1TransOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `ldmatrix_x1`: Load one 8x8 matrix tile from shared memory.
///
/// Args:
/// - `args[0]`: `*const u32` - Source pointer in shared memory
///
/// Returns: `u32` (single register with 2 packed b16 values)
pub fn emit_ldmatrix_x1(
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
                "ldmatrix_x1 expects 1 argument (smem_ptr), got {}",
                args.len()
            ))
        );
    }

    let (smem_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let ld_op = Operation::new(
        ctx,
        LdmatrixX1Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![smem_val],
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    let result = ld_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        ld_op,
        value_map,
        block_map,
        loc,
        "ldmatrix_x1 call without target block",
    )
}

/// Emit `ldmatrix_x1_trans`: Load one 8x8 matrix tile from shared memory with transpose.
///
/// Args:
/// - `args[0]`: `*const u32` - Source pointer in shared memory
///
/// Returns: `u32` (single register with 2 packed b16 values, transposed)
pub fn emit_ldmatrix_x1_trans(
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
                "ldmatrix_x1_trans expects 1 argument (smem_ptr), got {}",
                args.len()
            ))
        );
    }

    let (smem_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let ld_op = Operation::new(
        ctx,
        LdmatrixX1TransOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![smem_val],
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    let result = ld_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        ld_op,
        value_map,
        block_map,
        loc,
        "ldmatrix_x1_trans call without target block",
    )
}
