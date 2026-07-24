/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar floating-point intrinsics.

include!("generated/float.rs");

// =========================================================================
// Rounding intrinsics (single-instruction PTX via ptx_asm!)
// =========================================================================

/// Rounds `x` toward minus infinity (floor).
///
/// Maps to PTX: `cvt.rmi.f32.f32` (single instruction).
///
/// More efficient than routing through libdevice's `__nv_floorf`.
#[must_use]
#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    let result: f32;
    unsafe {
        crate::ptx_asm!(
            "cvt.rmi.f32.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` toward minus infinity (floor).
///
/// Maps to PTX: `cvt.rmi.f64.f64` (single instruction).
///
/// More efficient than routing through libdevice's `__nv_floor`.
#[must_use]
#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
    let result: f64;
    unsafe {
        crate::ptx_asm!(
            "cvt.rmi.f64.f64 %0, %1;",
            out("=d") result,
            in("d") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` toward plus infinity (ceiling).
///
/// Maps to PTX: `cvt.rpi.f32.f32` (single instruction).
///
/// More efficient than routing through libdevice's `__nv_ceilf`.
#[must_use]
#[inline(always)]
pub fn ceil_f32(x: f32) -> f32 {
    let result: f32;
    unsafe {
        crate::ptx_asm!(
            "cvt.rpi.f32.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` toward plus infinity (ceiling).
///
/// Maps to PTX: `cvt.rpi.f64.f64` (single instruction).
///
/// More efficient than routing through libdevice's `__nv_ceil`.
#[must_use]
#[inline(always)]
pub fn ceil_f64(x: f64) -> f64 {
    let result: f64;
    unsafe {
        crate::ptx_asm!(
            "cvt.rpi.f64.f64 %0, %1;",
            out("=d") result,
            in("d") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` toward zero (truncation).
///
/// Maps to PTX: `cvt.rzi.f32.f32` (single instruction).
///
/// More efficient than routing through libdevice's `__nv_truncf`.
#[must_use]
#[inline(always)]
pub fn trunc_f32(x: f32) -> f32 {
    let result: f32;
    unsafe {
        crate::ptx_asm!(
            "cvt.rzi.f32.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` toward zero (truncation).
///
/// Maps to PTX: `cvt.rzi.f64.f64` (single instruction).
///
/// More efficient than routing through libdevice's `__nv_trunc`.
#[must_use]
#[inline(always)]
pub fn trunc_f64(x: f64) -> f64 {
    let result: f64;
    unsafe {
        crate::ptx_asm!(
            "cvt.rzi.f64.f64 %0, %1;",
            out("=d") result,
            in("d") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` to the nearest even integer (IEEE 754 default rounding).
///
/// Maps to PTX: `cvt.rni.f32.f32` (single instruction).
///
/// This is the IEEE 754 "round to nearest, ties to even" mode, also known
/// as banker's rounding.
#[must_use]
#[inline(always)]
pub fn roundeven_f32(x: f32) -> f32 {
    let result: f32;
    unsafe {
        crate::ptx_asm!(
            "cvt.rni.f32.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// Rounds `x` to the nearest even integer (IEEE 754 default rounding).
///
/// Maps to PTX: `cvt.rni.f64.f64` (single instruction).
///
/// This is the IEEE 754 "round to nearest, ties to even" mode, also known
/// as banker's rounding.
#[must_use]
#[inline(always)]
pub fn roundeven_f64(x: f64) -> f64 {
    let result: f64;
    unsafe {
        crate::ptx_asm!(
            "cvt.rni.f64.f64 %0, %1;",
            out("=d") result,
            in("d") x,
            options(register_only),
        );
    }
    result
}
