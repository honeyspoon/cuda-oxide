/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hardware approximate math intrinsics.
//!
//! These map directly to single-cycle PTX approximate instructions, bypassing
//! multi-instruction libdevice paths. Useful in latency-sensitive code such as
//! activation functions and fast reciprocals.
//!
//! | Function | PTX instruction | Min SM |
//! |---|---|---|
//! | [`tanh_approx_f32`] | `tanh.approx.f32` | sm_75 |
//! | [`ex2_approx_ftz_f32`] | `ex2.approx.ftz.f32` | all |
//! | [`rcp_approx_ftz_f32`] | `rcp.approx.ftz.f32` | all |
//! | [`lg2_approx_ftz_f32`] | `lg2.approx.ftz.f32` | all |

/// Hardware approximate hyperbolic tangent (single-cycle).
///
/// Maps to PTX: `tanh.approx.f32`
///
/// Requires sm_75 or later. The result is an approximation — not IEEE-correct.
/// Subnormal inputs are flushed to zero.
#[must_use]
#[inline(never)]
pub fn tanh_approx_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!(
        "CUDA intrinsic `cuda_device::approx::tanh_approx_f32` \
         executed outside device compilation"
    )
}

/// Hardware approximate base-2 exponential (single-cycle, flush-to-zero).
///
/// Maps to PTX: `ex2.approx.ftz.f32`
///
/// Available on all SM architectures. Subnormal inputs are flushed to zero.
#[must_use]
#[inline(never)]
pub fn ex2_approx_ftz_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!(
        "CUDA intrinsic `cuda_device::approx::ex2_approx_ftz_f32` \
         executed outside device compilation"
    )
}

/// Hardware approximate reciprocal (single-cycle, flush-to-zero).
///
/// Maps to PTX: `rcp.approx.ftz.f32`
///
/// Available on all SM architectures. Subnormal inputs are flushed to zero.
#[must_use]
#[inline(never)]
pub fn rcp_approx_ftz_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!(
        "CUDA intrinsic `cuda_device::approx::rcp_approx_ftz_f32` \
         executed outside device compilation"
    )
}

/// Hardware approximate base-2 logarithm (single-cycle, flush-to-zero).
///
/// Maps to PTX: `lg2.approx.ftz.f32`
///
/// Available on all SM architectures. Subnormal inputs are flushed to zero.
#[must_use]
#[inline(never)]
pub fn lg2_approx_ftz_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!(
        "CUDA intrinsic `cuda_device::approx::lg2_approx_ftz_f32` \
         executed outside device compilation"
    )
}
