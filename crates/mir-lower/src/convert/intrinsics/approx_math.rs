/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering for approximate math operations → LLVM inline PTX.

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::{
    builtin::types::FP32Type,
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::DialectConversionRewriter, inserter::Inserter, rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
};

/// Lower a unary approximate math op to inline PTX.
///
/// All operations in this module share the same shape: one f32 operand,
/// one f32 result, pure (no side effects), and a single PTX instruction:
///
/// ```text
/// {ptx_mnemonic} $0, $1;
/// ```
pub(crate) fn convert_approx_math_unary(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "approximate math lowering requires exactly one operand and one result"
        );
    }

    let result_ty = FP32Type::get(ctx);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_mnemonic} $0, $1;"),
        "=f,f",
        AsmKind::Pure,
    );
    let inline_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, inline_op);
    rewriter.replace_operation(ctx, op, inline_op);
    Ok(())
}
