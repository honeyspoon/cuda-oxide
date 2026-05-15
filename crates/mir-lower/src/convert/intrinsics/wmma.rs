/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (mma.sync) intrinsic conversion for Ampere+ GPUs.
//!
//! Converts dialect-nvvm WMMA operations into inline PTX assembly.
//!
//! # Operations
//!
//! | Operation            | PTX                                              |
//! |----------------------|--------------------------------------------------|
//! | `LdmatrixX4`         | `ldmatrix.sync.aligned.m8n8.x4.shared.b16`      |
//! | `LdmatrixX2`         | `ldmatrix.sync.aligned.m8n8.x2.shared.b16`      |
//! | `LdmatrixX4Trans`    | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` |
//! | `LdmatrixX2Trans`    | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` |
//! | `MmaM16N8K16F32F16`  | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |

use crate::convert::intrinsics::common::*;
use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Shared implementation for all ldmatrix lowering variants.
///
/// Builds inline PTX for `ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16`
/// that loads `num_regs` × u32 from shared memory and stores to `dest_ptr`.
///
/// Note: `smem_ptr` is a generic-space pointer. The PTX uses `cvta.to.shared`
/// to convert it (same pattern as stmatrix.rs). Do NOT use
/// `cast_to_shared_addrspace` — that would double-convert.
fn convert_ldmatrix_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    num_regs: usize,
    trans: bool,
    name: &str,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("{} requires 2 operands (smem_ptr, dest_ptr)", name);
    }
    let smem_ptr = operands[0];
    let dest_ptr = operands[1];

    // Build register list: {r0} or {r0, r1} or {r0, r1, r2, r3}
    let reg_list: String = (0..num_regs)
        .map(|i| format!("r{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let trans_suffix = if trans { ".trans" } else { "" };

    // Build store sequence: st.b32 [$0+offset], rN;
    let stores: String = (0..num_regs)
        .map(|i| {
            if i == 0 {
                format!("st.b32 [$0], r0; ")
            } else {
                format!("st.b32 [$0+{}], r{i}; ", i * 4)
            }
        })
        .collect::<String>();

    let asm = format!(
        "{{ \
         .reg .b32 r<{num_regs}>; \
         .reg .u64 smem64; \
         .reg .u32 smem32; \
         cvta.to.shared.u64 smem64, $1; \
         cvt.u32.u64 smem32, smem64; \
         ldmatrix.sync.aligned.m8n8.x{num_regs}{trans_suffix}.shared.b16 {{{reg_list}}}, [smem32]; \
         {stores}\
         }}"
    );

    inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![dest_ptr, smem_ptr], &asm, "l,l");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `ldmatrix.sync.aligned.m8n8.x4.shared.b16` — load 4 × u32 from shared.
pub(crate) fn convert_ldmatrix_x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 4, false, "ldmatrix_x4")
}

/// Convert `ldmatrix.sync.aligned.m8n8.x2.shared.b16` — load 2 × u32 from shared.
pub(crate) fn convert_ldmatrix_x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 2, false, "ldmatrix_x2")
}

/// Convert `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` — load 4 × u32 transposed.
pub(crate) fn convert_ldmatrix_x4_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 4, true, "ldmatrix_x4_trans")
}

/// Convert `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` — load 2 × u32 transposed.
pub(crate) fn convert_ldmatrix_x2_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 2, true, "ldmatrix_x2_trans")
}

/// Convert mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///
/// Operands: [acc_ptr, a_ptr, b_ptr]
/// - acc_ptr: pointer to [f32; 4] (read-modify-write)
/// - a_ptr:   pointer to [u32; 4] (A fragment)
/// - b_ptr:   pointer to [u32; 2] (B fragment)
///
/// The lowering loads the fragments from pointers into PTX registers,
/// executes the mma.sync instruction, and stores results back.
/// Uses generic ld/st since the pointers are in generic address space.
pub(crate) fn convert_mma_m16n8k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_f32_f16 requires 3 operands (acc_ptr, a_ptr, b_ptr)"
        );
    }
    let acc_ptr = operands[0];
    let a_ptr = operands[1];
    let b_ptr = operands[2];

    let asm = concat!(
        "{ ",
        ".reg .f32 d<4>; ",
        ".reg .b32 a<4>; ",
        ".reg .b32 b<2>; ",
        "ld.f32 d0, [$0]; ",
        "ld.f32 d1, [$0+4]; ",
        "ld.f32 d2, [$0+8]; ",
        "ld.f32 d3, [$0+12]; ",
        "ld.b32 a0, [$1]; ",
        "ld.b32 a1, [$1+4]; ",
        "ld.b32 a2, [$1+8]; ",
        "ld.b32 a3, [$1+12]; ",
        "ld.b32 b0, [$2]; ",
        "ld.b32 b1, [$2+4]; ",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{d0, d1, d2, d3}, ",
        "{a0, a1, a2, a3}, ",
        "{b0, b1}, ",
        "{d0, d1, d2, d3}; ",
        "st.f32 [$0], d0; ",
        "st.f32 [$0+4], d1; ",
        "st.f32 [$0+8], d2; ",
        "st.f32 [$0+12], d3; ",
        "}"
    );

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![acc_ptr, a_ptr, b_ptr],
        asm,
        "l,l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert fused K-step: ldmatrix_x4(A) + 4×ldmatrix_x2_trans(B) + 4×mma.sync
///
/// Operands: [a_smem, b_smem0, b_smem1, b_smem2, b_smem3, acc0, acc1, acc2, acc3]
///
/// Uses InlineAsmMultiOp with tied register constraints so that accumulators
/// live in PTX registers, not local memory. LLVM loads/stores the accumulator
/// values outside the asm block, enabling mem2reg optimization across K-steps.
pub(crate) fn convert_wmma_fused_k_step_4x(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 9 {
        return pliron::input_err_noloc!(
            "wmma_fused_k_step_4x requires 9 operands"
        );
    }
    let a_smem = operands[0];
    let b_smem0 = operands[1];
    let b_smem1 = operands[2];
    let b_smem2 = operands[3];
    let b_smem3 = operands[4];
    let acc0_ptr = operands[5];
    let acc1_ptr = operands[6];
    let acc2_ptr = operands[7];
    let acc3_ptr = operands[8];

    let f32_ty: Ptr<pliron::r#type::TypeObj> = FP32Type::get(ctx).into();

    // Helper: load f32 at ptr + byte_offset
    // Uses GEP on f32 element type (so index is in f32 units, not bytes)
    let load_f32 = |ctx: &mut Context,
                    rewriter: &mut DialectConversionRewriter,
                    ptr: pliron::value::Value,
                    elem_idx: u32|
     -> pliron::value::Value {
        if elem_idx == 0 {
            let ld = llvm::LoadOp::new(ctx, ptr, f32_ty);
            rewriter.insert_operation(ctx, ld.get_operation());
            ld.get_operation().deref(ctx).get_result(0)
        } else {
            let gep = llvm::GetElementPtrOp::new(
                ctx,
                ptr,
                vec![llvm::GepIndex::Constant(elem_idx)],
                f32_ty,
            )
            .unwrap();
            rewriter.insert_operation(ctx, gep.get_operation());
            let elem_ptr = gep.get_operation().deref(ctx).get_result(0);
            let ld = llvm::LoadOp::new(ctx, elem_ptr, f32_ty);
            rewriter.insert_operation(ctx, ld.get_operation());
            ld.get_operation().deref(ctx).get_result(0)
        }
    };

    // Helper: store f32 at ptr + byte_offset
    let store_f32 = |ctx: &mut Context,
                     rewriter: &mut DialectConversionRewriter,
                     ptr: pliron::value::Value,
                     val: pliron::value::Value,
                     elem_idx: u32| {
        if elem_idx == 0 {
            let st = llvm::StoreOp::new(ctx, val, ptr);
            rewriter.insert_operation(ctx, st.get_operation());
        } else {
            let gep = llvm::GetElementPtrOp::new(
                ctx,
                ptr,
                vec![llvm::GepIndex::Constant(elem_idx)],
                f32_ty,
            )
            .unwrap();
            rewriter.insert_operation(ctx, gep.get_operation());
            let elem_ptr = gep.get_operation().deref(ctx).get_result(0);
            let st = llvm::StoreOp::new(ctx, val, elem_ptr);
            rewriter.insert_operation(ctx, st.get_operation());
        }
    };

    // Load 16 f32 accumulator values from pointers
    let acc_ptrs = [acc0_ptr, acc1_ptr, acc2_ptr, acc3_ptr];
    let mut acc_vals = Vec::with_capacity(16);
    for &acc_ptr in &acc_ptrs {
        for j in 0..4u32 {
            acc_vals.push(load_f32(ctx, rewriter, acc_ptr, j));
        }
    }

    // PTX asm template using register operands.
    // Operand layout (InlineAsmMultiOp::new_tied_convergent):
    //   $0..$15:  16 tied f32 outputs (accumulators)
    //   $16..$31: 16 tied f32 inputs (same regs as $0..$15)
    //   $32..$36: 5 smem pointer inputs (l constraints)
    let asm = concat!(
        "{ ",
        ".reg .b32 a<4>; ",
        ".reg .b32 b<2>; ",
        ".reg .u64 smem64; ",
        ".reg .u32 smem32; ",

        // ldmatrix_x4 for A tile (smem ptr = $32)
        "cvta.to.shared.u64 smem64, $32; ",
        "cvt.u32.u64 smem32, smem64; ",
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0, a1, a2, a3}, [smem32]; ",

        // B0 + mma0 (smem ptr = $33, acc = $0-$3)
        "cvta.to.shared.u64 smem64, $33; ",
        "cvt.u32.u64 smem32, smem64; ",
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {b0, b1}, [smem32]; ",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{$0, $1, $2, $3}, ",
        "{a0, a1, a2, a3}, ",
        "{b0, b1}, ",
        "{$0, $1, $2, $3}; ",

        // B1 + mma1 (smem ptr = $34, acc = $4-$7)
        "cvta.to.shared.u64 smem64, $34; ",
        "cvt.u32.u64 smem32, smem64; ",
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {b0, b1}, [smem32]; ",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{$4, $5, $6, $7}, ",
        "{a0, a1, a2, a3}, ",
        "{b0, b1}, ",
        "{$4, $5, $6, $7}; ",

        // B2 + mma2 (smem ptr = $35, acc = $8-$11)
        "cvta.to.shared.u64 smem64, $35; ",
        "cvt.u32.u64 smem32, smem64; ",
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {b0, b1}, [smem32]; ",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{$8, $9, $10, $11}, ",
        "{a0, a1, a2, a3}, ",
        "{b0, b1}, ",
        "{$8, $9, $10, $11}; ",

        // B3 + mma3 (smem ptr = $36, acc = $12-$15)
        "cvta.to.shared.u64 smem64, $36; ",
        "cvt.u32.u64 smem32, smem64; ",
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {b0, b1}, [smem32]; ",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{$12, $13, $14, $15}, ",
        "{a0, a1, a2, a3}, ",
        "{b0, b1}, ",
        "{$12, $13, $14, $15}; ",
        "}"
    );

    // Create InlineAsmMultiOp with 16 tied f32 + 5 smem pointer inputs
    let smem_inputs = vec![a_smem, b_smem0, b_smem1, b_smem2, b_smem3];

    let multi_asm = llvm::InlineAsmMultiOp::new_tied_convergent(
        ctx,
        16,       // num_tied (16 f32 accumulators)
        f32_ty,   // output type for each tied operand
        acc_vals, // tied inputs (16 f32 values)
        smem_inputs,
        asm,
        "f",      // output constraint for f32 register
        "l,l,l,l,l",  // 5 smem pointers (no memory clobber: sync_threads provides ordering)
    );
    rewriter.insert_operation(ctx, multi_asm.get_operation());

    // Store 16 f32 results back to accumulator pointers
    let asm_op = multi_asm.get_operation();
    for (i, &acc_ptr) in acc_ptrs.iter().enumerate() {
        for j in 0..4u32 {
            let result_idx = (i * 4 + j as usize) as usize;
            let result_val = asm_op.deref(ctx).get_result(result_idx);
            store_f32(ctx, rewriter, acc_ptr, result_val, j);
        }
    }

    rewriter.erase_operation(ctx, op);
    Ok(())
}
