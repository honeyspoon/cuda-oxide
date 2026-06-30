/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsic lowering (`movmatrix`).

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `nvvm.movmatrix_trans_b16` to inline PTX.
///
/// `movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;`
///
/// Warp-synchronous, uses convergent inline assembly.
pub(crate) fn convert_movmatrix_trans_b16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "movmatrix_trans_b16 requires 1 operand, got {}",
            operands.len()
        );
    }

    let a_val = operands[0];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        AsmKind::Convergent,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `mma_m16n8k8_f32_bf16` to inline PTX assembly.
pub(crate) fn convert_mma_m16n8k8_f32_bf16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_sync(
        ctx,
        rewriter,
        op,
        ".reg .f32 c<4>; .reg .f32 d<4>; .reg .b32 a<2>; .reg .b32 b0; ld.f32 c0, [$0]; ld.f32 c1, [$0+4]; ld.f32 c2, [$0+8]; ld.f32 c3, [$0+12]; ld.b32 a0, [$1]; ld.b32 a1, [$1+4]; ld.b32 b0, [$2]; mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {d0, d1, d2, d3}, {a0, a1}, {b0}, {c0, c1, c2, c3}; st.f32 [$0], d0; st.f32 [$0+4], d1; st.f32 [$0+8], d2; st.f32 [$0+12], d3;",
        "l,l,l,~{memory}",
        3,
        "mma_m16n8k8_f32_bf16",
    )
}

/// Convert `mma_m16n8k4_f32_tf32` to inline PTX assembly.
pub(crate) fn convert_mma_m16n8k4_f32_tf32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_sync(
        ctx,
        rewriter,
        op,
        ".reg .f32 c<4>; .reg .f32 d<4>; .reg .b32 a<2>; .reg .b32 b0; ld.f32 c0, [$0]; ld.f32 c1, [$0+4]; ld.f32 c2, [$0+8]; ld.f32 c3, [$0+12]; ld.b32 a0, [$1]; ld.b32 a1, [$1+4]; ld.b32 b0, [$2]; mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {d0, d1, d2, d3}, {a0, a1}, {b0}, {c0, c1, c2, c3}; st.f32 [$0], d0; st.f32 [$0+4], d1; st.f32 [$0+8], d2; st.f32 [$0+12], d3;",
        "l,l,l,~{memory}",
        3,
        "mma_m16n8k4_f32_tf32",
    )
}