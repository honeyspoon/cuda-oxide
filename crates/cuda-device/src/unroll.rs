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
    // Non-power-of-2 range forms: delegate to nearest power-of-2 + remainder
    ($i:ident in 0..5 => $body:block) => {
        $crate::unroll!($i in 0..4 => $body);
        { const $i: usize = 4; $body }
    };
    ($i:ident in 0..6 => $body:block) => {
        $crate::unroll!($i in 0..4 => $body);
        { const $i: usize = 4; $body }
        { const $i: usize = 5; $body }
    };
    ($i:ident in 0..7 => $body:block) => {
        $crate::unroll!($i in 0..4 => $body);
        { const $i: usize = 4; $body }
        { const $i: usize = 5; $body }
        { const $i: usize = 6; $body }
    };
    ($i:ident in 0..10 => $body:block) => {
        $crate::unroll!($i in 0..8 => $body);
        { const $i: usize = 8; $body }
        { const $i: usize = 9; $body }
    };
    ($i:ident in 0..12 => $body:block) => {
        $crate::unroll!($i in 0..8 => $body);
        { const $i: usize = 8; $body }
        { const $i: usize = 9; $body }
        { const $i: usize = 10; $body }
        { const $i: usize = 11; $body }
    };
    ($i:ident in 0..24 => $body:block) => {
        $crate::unroll!($i in 0..16 => $body);
        { const $i: usize = 16; $body }
        { const $i: usize = 17; $body }
        { const $i: usize = 18; $body }
        { const $i: usize = 19; $body }
        { const $i: usize = 20; $body }
        { const $i: usize = 21; $body }
        { const $i: usize = 22; $body }
        { const $i: usize = 23; $body }
    };
    ($i:ident in 0..64 => $body:block) => {
        $crate::unroll!($i in 0..32 => $body);
        { const $i: usize = 32; $body }
        { const $i: usize = 33; $body }
        { const $i: usize = 34; $body }
        { const $i: usize = 35; $body }
        { const $i: usize = 36; $body }
        { const $i: usize = 37; $body }
        { const $i: usize = 38; $body }
        { const $i: usize = 39; $body }
        { const $i: usize = 40; $body }
        { const $i: usize = 41; $body }
        { const $i: usize = 42; $body }
        { const $i: usize = 43; $body }
        { const $i: usize = 44; $body }
        { const $i: usize = 45; $body }
        { const $i: usize = 46; $body }
        { const $i: usize = 47; $body }
        { const $i: usize = 48; $body }
        { const $i: usize = 49; $body }
        { const $i: usize = 50; $body }
        { const $i: usize = 51; $body }
        { const $i: usize = 52; $body }
        { const $i: usize = 53; $body }
        { const $i: usize = 54; $body }
        { const $i: usize = 55; $body }
        { const $i: usize = 56; $body }
        { const $i: usize = 57; $body }
        { const $i: usize = 58; $body }
        { const $i: usize = 59; $body }
        { const $i: usize = 60; $body }
        { const $i: usize = 61; $body }
        { const $i: usize = 62; $body }
        { const $i: usize = 63; $body }
    };
    // List form: unroll!(i in [a, b, c] => { body })
    ($i:ident in [$($val:expr),+ $(,)?] => $body:block) => {
        $( { const $i: usize = $val; $body } )+
    };
}
