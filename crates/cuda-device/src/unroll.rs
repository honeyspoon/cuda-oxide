/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-time loop unrolling macros for GPU kernels.
//!
//! GPU performance depends heavily on instruction-level parallelism.
//! LLVM's loop unroller doesn't always unroll loops with complex bounds
//! (e.g., `while ki + 16 <= kk`), even when the trip count is known at
//! compile time. These macros guarantee unrolling by generating the loop
//! body N times at compile time.
//!
//! # Example
//!
//! ```rust,ignore
//! use cuda_device::unroll;
//!
//! // Unroll a 4-iteration load
//! unroll!(i in 0..4 => {
//!     let val = unsafe { ptr.add(i).read() };
//!     acc += val;
//! });
//!
//! // Unroll with step
//! unroll!(i in [0, 4, 8, 12] => {
//!     process_tile(i);
//! });
//! ```

/// Compile-time loop unrolling macro.
///
/// Generates N copies of the body with the loop variable bound to
/// successive values. The variable is a `usize` constant expression
/// visible in the body.
///
/// # Supported forms
///
/// ```rust,ignore
/// // Range form: unrolls for i = 0, 1, 2, 3
/// unroll!(i in 0..4 => { body });
///
/// // List form: unrolls for each listed value
/// unroll!(i in [0, 2, 4, 6] => { body });
/// ```
///
/// # Why not `#[unroll]`?
///
/// The `#[unroll]` attribute relies on LLVM's loop unroller, which uses
/// heuristics that can fail for loops with:
/// - Complex bounds (`while ki + 16 <= kk`)
/// - Data-dependent exit conditions
/// - Large loop bodies that exceed the unroller's threshold
///
/// This macro guarantees unrolling by generating the code at compile time.
#[macro_export]
macro_rules! unroll {
    // Range form: unroll!(i in 0..N => { body })
    ($i:ident in 0..1 => $body:block) => {
        { const $i: usize = 0; $body }
    };
    ($i:ident in 0..2 => $body:block) => {
        { const $i: usize = 0; $body }
        { const $i: usize = 1; $body }
    };
    ($i:ident in 0..3 => $body:block) => {
        { const $i: usize = 0; $body }
        { const $i: usize = 1; $body }
        { const $i: usize = 2; $body }
    };
    ($i:ident in 0..4 => $body:block) => {
        { const $i: usize = 0; $body }
        { const $i: usize = 1; $body }
        { const $i: usize = 2; $body }
        { const $i: usize = 3; $body }
    };
    ($i:ident in 0..8 => $body:block) => {
        { const $i: usize = 0; $body }
        { const $i: usize = 1; $body }
        { const $i: usize = 2; $body }
        { const $i: usize = 3; $body }
        { const $i: usize = 4; $body }
        { const $i: usize = 5; $body }
        { const $i: usize = 6; $body }
        { const $i: usize = 7; $body }
    };
    ($i:ident in 0..16 => $body:block) => {
        $crate::unroll!($i in 0..8 => $body);
        { const $i: usize = 8; $body }
        { const $i: usize = 9; $body }
        { const $i: usize = 10; $body }
        { const $i: usize = 11; $body }
        { const $i: usize = 12; $body }
        { const $i: usize = 13; $body }
        { const $i: usize = 14; $body }
        { const $i: usize = 15; $body }
    };
    ($i:ident in 0..32 => $body:block) => {
        $crate::unroll!($i in 0..16 => $body);
        { const $i: usize = 16; $body }
        { const $i: usize = 17; $body }
        { const $i: usize = 18; $body }
        { const $i: usize = 19; $body }
        { const $i: usize = 20; $body }
        { const $i: usize = 21; $body }
        { const $i: usize = 22; $body }
        { const $i: usize = 23; $body }
        { const $i: usize = 24; $body }
        { const $i: usize = 25; $body }
        { const $i: usize = 26; $body }
        { const $i: usize = 27; $body }
        { const $i: usize = 28; $body }
        { const $i: usize = 29; $body }
        { const $i: usize = 30; $body }
        { const $i: usize = 31; $body }
    };
    // List form: unroll!(i in [a, b, c] => { body })
    ($i:ident in [$($val:expr),+ $(,)?] => $body:block) => {
        $( { const $i: usize = $val; $body } )+
    };
}
