/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar floating-point intrinsics.

include!("generated/float.rs");

// =========================================================================
// Rounding intrinsics (single-instruction PTX via ptx_asm!)
// =========================================================================
//
// PTX exposes exactly four float-to-integer rounding modes as a single `cvt`
// instruction, and all four are wrapped below:
//
//   cvt.rmi -> toward minus infinity   (floor)
//   cvt.rpi -> toward plus infinity    (ceil)
//   cvt.rzi -> toward zero             (trunc)
//   cvt.rni -> to nearest, ties to even (roundeven)
//
// Ties away from zero is deliberately absent: it is not one of the hardware
// modes, so it cannot be a single instruction. `roundeven_f32` is not a
// substitute -- the two disagree at every exact `.5` tie (2.5 rounds to 2.0
// under ties-to-even but 3.0 under ties-away), and the difference is silent.
// Callers that need ties-away must keep their own sequence; correcting
// `roundeven` at ties, or `trunc(x + copysign(0.5, x))` guarded for
// magnitudes at or above 2^23 where every f32 is already an integer and the
// addition would round.
//
// Relationship to the compiler path: once floor/ceil/trunc/round_ties_even
// are routed onto LLVM intrinsics in mir-lower, `f32::floor()` and friends
// lower to these same `cvt` instructions, and the wrappers here become a way
// to name the rounding mode explicitly rather than a performance win.

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
