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
/// successive values. The variable is bound as a `const`, so it is usable
/// in const position (array lengths, const generic arguments), not just as
/// a runtime value.
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
/// Supported range bounds are `0..N` for N in 1-32, plus the common
/// non-power-of-2 tile factors 5, 6, 7, 10, 12, 24, and 64. Other bounds
/// do not match any arm and produce a compile error; use the list form.
///
/// # Lint suppression
///
/// Loop variables are conventionally lower-case (`i`, `j`), but the binding
/// expands to a `const` item, which `non_upper_case_globals` would flag at
/// every call site. Because the workspace builds with `-D warnings`, that
/// would make the macro unusable. Each expansion therefore carries its own
/// `#[allow(non_upper_case_globals)]` so callers need no attribute.
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
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body }
    };
    ($i:ident in 0..2 => $body:block) => {
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 1; $body }
    };
    ($i:ident in 0..3 => $body:block) => {
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 1; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 2; $body }
    };
    ($i:ident in 0..4 => $body:block) => {
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 1; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 2; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 3; $body }
    };
    ($i:ident in 0..8 => $body:block) => {
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 1; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 2; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 3; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 4; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 5; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 6; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 7; $body }
    };
    ($i:ident in 0..16 => $body:block) => {
        $crate::unroll!($i in 0..8 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 8; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 9; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 10; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 11; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 12; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 13; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 14; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 15; $body }
    };
    ($i:ident in 0..32 => $body:block) => {
        $crate::unroll!($i in 0..16 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 16; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 17; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 18; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 19; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 20; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 21; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 22; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 23; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 24; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 25; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 26; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 27; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 28; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 29; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 30; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 31; $body }
    };
    // Non-power-of-2 range forms: delegate to nearest power-of-2 + remainder
    ($i:ident in 0..5 => $body:block) => {
        $crate::unroll!($i in 0..4 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 4; $body }
    };
    ($i:ident in 0..6 => $body:block) => {
        $crate::unroll!($i in 0..4 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 4; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 5; $body }
    };
    ($i:ident in 0..7 => $body:block) => {
        $crate::unroll!($i in 0..4 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 4; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 5; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 6; $body }
    };
    ($i:ident in 0..10 => $body:block) => {
        $crate::unroll!($i in 0..8 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 8; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 9; $body }
    };
    ($i:ident in 0..12 => $body:block) => {
        $crate::unroll!($i in 0..8 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 8; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 9; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 10; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 11; $body }
    };
    ($i:ident in 0..24 => $body:block) => {
        $crate::unroll!($i in 0..16 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 16; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 17; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 18; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 19; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 20; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 21; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 22; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 23; $body }
    };
    ($i:ident in 0..64 => $body:block) => {
        $crate::unroll!($i in 0..32 => $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 32; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 33; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 34; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 35; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 36; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 37; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 38; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 39; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 40; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 41; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 42; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 43; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 44; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 45; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 46; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 47; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 48; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 49; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 50; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 51; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 52; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 53; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 54; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 55; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 56; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 57; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 58; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 59; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 60; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 61; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 62; $body }
        { #[allow(non_upper_case_globals)] const $i: usize = 63; $body }
    };
    // List form: unroll!(i in [a, b, c] => { body })
    ($i:ident in [$($val:expr),+ $(,)?] => $body:block) => {
        $( { #[allow(non_upper_case_globals)] const $i: usize = $val; $body } )+
    };
}

#[cfg(test)]
mod tests {
    //! The macro expands to `const` items, so these tests also assert that no
    //! expansion trips `non_upper_case_globals` under the workspace's
    //! `-D warnings` build.

    /// Sum of `0..n`, used as the expected value for each range arm.
    const fn triangular(n: usize) -> usize {
        n * (n - 1) / 2
    }

    #[test]
    fn range_arms_cover_every_index() {
        // Power-of-2 arms.
        let mut got = 0;
        unroll!(i in 0..1 => { got += i + 1; });
        assert_eq!(got, 1, "0..1 must run exactly once");

        let mut sum4 = 0;
        unroll!(i in 0..4 => { sum4 += i; });
        assert_eq!(sum4, triangular(4));

        let mut sum32 = 0;
        unroll!(i in 0..32 => { sum32 += i; });
        assert_eq!(sum32, triangular(32));
    }

    #[test]
    fn non_power_of_two_arms_delegate_without_gaps() {
        // These arms delegate to the nearest power-of-2 arm and then emit the
        // remainder inline; a wrong split shows up as a wrong sum.
        let mut sum5 = 0;
        unroll!(i in 0..5 => { sum5 += i; });
        assert_eq!(sum5, triangular(5));

        let mut sum7 = 0;
        unroll!(i in 0..7 => { sum7 += i; });
        assert_eq!(sum7, triangular(7));

        let mut sum12 = 0;
        unroll!(i in 0..12 => { sum12 += i; });
        assert_eq!(sum12, triangular(12));

        let mut sum24 = 0;
        unroll!(i in 0..24 => { sum24 += i; });
        assert_eq!(sum24, triangular(24));

        let mut sum64 = 0;
        unroll!(i in 0..64 => { sum64 += i; });
        assert_eq!(sum64, triangular(64));
    }

    #[test]
    fn list_form_visits_each_value_in_order() {
        let mut seen = [0usize; 4];
        let mut n = 0;
        unroll!(i in [0, 2, 4, 6] => {
            seen[n] = i;
            n += 1;
        });
        assert_eq!(n, 4);
        assert_eq!(seen, [0, 2, 4, 6]);
    }

    #[test]
    fn loop_variable_is_usable_in_const_position() {
        // Binding as `const` (not `let`) is what makes this work; a `let`
        // binding would not be accepted as an array length.
        let mut total = 0;
        unroll!(n in 0..4 => {
            let buf = [7u32; n];
            total += buf.len();
        });
        assert_eq!(total, triangular(4));
    }
}
