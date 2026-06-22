/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ldmatrix intrinsic conversion for matrix load operations.
//!
//! # Operations
//!
//! | Operation     | PTX                                             | Description         |
//! |---------------|-------------------------------------------------|---------------------|
//! | `X1`          | `ldmatrix.sync.aligned.m8n8.x1.shared.b16`      | Load 1 8x8 matrix   |
//! | `X1Trans`     | `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16`| Load 1 transposed   |

use crate::convert::intrinsics::common::*;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Shared implementation for all ldmatrix variants.
///
/// Generates inline PTX assembly for `ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16`.
///
/// # Parameters
///
/// - `num_regs`: number of output registers (1 for x1)
/// - `trans`: whether to use the `.trans` modifier
fn convert_ldmatrix_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    num_regs: usize,
    trans: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("ldmatrix requires at least 1 operand (smem_ptr)");
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    // Build the output register list: {$0} for x1
    let reg_list: String = (0..num_regs)
        .map(|i| format!("${}", i))
        .collect::<Vec<_>>()
        .join(", ");

    // The pointer operand is after the output registers in the constraint numbering.
    let ptr_operand = format!("${}", num_regs);

    let trans_modifier = if trans { ".trans" } else { "" };

    let asm_template = format!(
        concat!(
            "{{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, {ptr}; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "ldmatrix.sync.aligned.m8n8.x{num}{trans}.shared.b16 {{{regs}}}, [%ptr32]; ",
            "}}"
        ),
        ptr = ptr_operand,
        num = num_regs,
        trans = trans_modifier,
        regs = reg_list,
    );

    // Build constraint string: "=r" for each output, then "l" for the pointer input.
    let output_constraints: String = (0..num_regs).map(|_| "=r").collect::<Vec<_>>().join(",");
    let constraints = format!("{},l", output_constraints);

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        operands,
        &asm_template,
        &constraints,
    );

    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

pub(crate) fn convert_ldmatrix_x1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 1, false)
}

pub(crate) fn convert_ldmatrix_x1_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 1, true)
}
