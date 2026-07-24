/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar floating-point intrinsics.

// =============================================================================
// Simple min/max
// =============================================================================
//
// Direct PTX `min.f32` / `max.f32` without the extended modifiers
// (xorsign_abs, nan propagation, etc.) that the generated variants use.
// These implement IEEE 754-2008 minNum/maxNum: if one operand is NaN, the
// non-NaN operand is returned.

/// Returns the smaller of two f32 values.
///
/// Maps to PTX: `min.f32`
///
/// If either operand is NaN, the non-NaN value is returned (IEEE 754
/// minNum semantics). If both are NaN, NaN is returned.
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
/// If either operand is NaN, the non-NaN value is returned (IEEE 754
/// maxNum semantics). If both are NaN, NaN is returned.
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
/// If either operand is NaN, the non-NaN value is returned (IEEE 754
/// minNum semantics). If both are NaN, NaN is returned.
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
/// If either operand is NaN, the non-NaN value is returned (IEEE 754
/// maxNum semantics). If both are NaN, NaN is returned.
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

include!("generated/float.rs");
