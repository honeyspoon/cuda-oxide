/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar floating-point intrinsics.

include!("generated/float.rs");

// =============================================================================
// Simple min/max
// =============================================================================
//
// Direct PTX `min` / `max` without the extended modifiers (`.xorsign.abs`,
// `.NaN`, `.ftz`) that the generated variants carry.
//
// NaN handling: plain `min`/`max` return the non-NaN operand when exactly one
// operand is NaN, and a canonical NaN when both are. Propagating NaN instead
// requires the `.NaN` modifier, which is what the generated `*_nan_*` variants
// use. That non-propagating behaviour is IEEE 754-2008 `minNum`/`maxNum`; the
// year matters, because 754-2019 withdrew those operations in favour of
// `minimum`/`maximum`, which do propagate NaN and are not what these emit.
//
// Why not `f32::min`: it lowers through libdevice (`__nv_fminf`). Rerouting it
// onto LLVM's `llvm.minnum`/`llvm.maxnum` is not a safe shortcut either, since
// under LLVM 21 those intrinsics propagate signaling NaNs and so disagree with
// the contract Rust documents for `f32::min`. Emitting the PTX instruction
// directly avoids both the call and that mismatch.

/// Returns the smaller of two f32 values.
///
/// Maps to PTX: `min.f32`
///
/// If exactly one operand is NaN, the other operand is returned; if both
/// are NaN, the result is a canonical NaN. Use the `*_nan_*` variants to
/// propagate NaN instead.
///
/// See also: [`min_xorsign_abs_f32`], [`min_nan_xorsign_abs_f32`]
#[must_use]
#[inline(always)]
pub fn fmin_f32(a: f32, b: f32) -> f32 {
    let result: f32;
    unsafe {
        crate::ptx_asm!(
            "min.f32 %0, %1, %2;",
            out("=f") result,
            in("f") a,
            in("f") b,
            options(register_only),
        );
    }
    result
}

/// Returns the larger of two f32 values.
///
/// Maps to PTX: `max.f32`
///
/// If exactly one operand is NaN, the other operand is returned; if both
/// are NaN, the result is a canonical NaN. Use the `*_nan_*` variants to
/// propagate NaN instead.
///
/// See also: [`max_xorsign_abs_f32`], [`max_nan_xorsign_abs_f32`]
#[must_use]
#[inline(always)]
pub fn fmax_f32(a: f32, b: f32) -> f32 {
    let result: f32;
    unsafe {
        crate::ptx_asm!(
            "max.f32 %0, %1, %2;",
            out("=f") result,
            in("f") a,
            in("f") b,
            options(register_only),
        );
    }
    result
}

/// Returns the smaller of two f64 values.
///
/// Maps to PTX: `min.f64`
///
/// If exactly one operand is NaN, the other operand is returned; if both
/// are NaN, the result is a canonical NaN. Use the `*_nan_*` variants to
/// propagate NaN instead.
///
/// The extended modifier forms (`.xorsign.abs`, `.NaN`, `.ftz`) are provided
/// for f32 only, so there is no f64 counterpart to cross-reference.
#[must_use]
#[inline(always)]
pub fn fmin_f64(a: f64, b: f64) -> f64 {
    let result: f64;
    unsafe {
        crate::ptx_asm!(
            "min.f64 %0, %1, %2;",
            out("=d") result,
            in("d") a,
            in("d") b,
            options(register_only),
        );
    }
    result
}

/// Returns the larger of two f64 values.
///
/// Maps to PTX: `max.f64`
///
/// If exactly one operand is NaN, the other operand is returned; if both
/// are NaN, the result is a canonical NaN. Use the `*_nan_*` variants to
/// propagate NaN instead.
///
/// The extended modifier forms (`.xorsign.abs`, `.NaN`, `.ftz`) are provided
/// for f32 only, so there is no f64 counterpart to cross-reference.
#[must_use]
#[inline(always)]
pub fn fmax_f64(a: f64, b: f64) -> f64 {
    let result: f64;
    unsafe {
        crate::ptx_asm!(
            "max.f64 %0, %1, %2;",
            out("=d") result,
            in("d") a,
            in("d") b,
            options(register_only),
        );
    }
    result
}
