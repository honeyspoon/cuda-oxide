/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit 32-bit index arithmetic, and a measurement of when it matters.
//!
//! `usize` is **64 bits** on the device: the codegen maps it that way for the
//! `nvptx64` target (`device_codegen.rs:188`). That is correct, and it looks
//! like a trap, because `usize` is what idiomatic Rust reaches for everywhere -
//! `slice::get` takes one, `<*const T>::add` takes one, and
//! `threadIdx_x() as usize` produces one. The natural worry is that a user does
//! the ordinary thing and silently pays for 64-bit arithmetic.
//!
//! # What the measurement actually shows
//!
//! Mostly, they do not pay, because `ptxas` narrows the arithmetic. Measured on
//! sm_86, SASS instruction counts:
//!
//! ```text
//! shape                                          usize    32-bit   delta
//! bounds-checked gather (cuda-oxide)               64        64       0
//! raw-pointer gather (cuda-oxide)                  48        48       0
//! runtime-derived index, no bound check            40        40       0
//! arith/compare width crossed four ways            32        32       0
//! ```
//!
//! At PTX level the difference is real but small - the `usize` gather emits 9
//! 64-bit arithmetic ops against 8 - and `ptxas` then erases it. Four of the
//! five shapes probed showed **no difference in final machine code**, including
//! both a bounds-checked and a raw-pointer version of the same kernel compiled
//! through cuda-oxide itself.
//!
//! The one shape that did differ (40 against 32) turned out to depend on which
//! expression indexed the *output*, not on the index width as such, and did not
//! survive being isolated. Treat it as noise rather than as a rule.
//!
//! # So what is this for
//!
//! Two things, neither of them "use this and go faster":
//!
//! 1. **Recording the negative result.** The 64-bit `usize` is real and easy to
//!    find alarming; the cost mostly is not. Anyone who notices the former
//!    should be able to find the latter without re-deriving it.
//! 2. **A tool for the cases where it does matter.** Complex index expressions
//!    that `ptxas` cannot narrow do exist, and profiling a specific kernel can
//!    show one. This type makes the arithmetic width explicit so that
//!    experiment is a small edit rather than a rewrite.
//!
//! **Measure before reaching for it.** On the evidence here the default
//! `usize` spelling is fine, and the burden is on a specific kernel to show
//! otherwise.
//!
//! # What would actually be worth changing
//!
//! In the bounds-checked kernel, the dominant cost is not index width at all:
//! four `BPT.TRAP` panic paths and nine `ISETP` bounds comparisons, against
//! four `LDG.E` that do the work. Bounds checking, not index arithmetic, is
//! what a safe indexing path pays for. That is gap #7's territory, not this
//! module's.
//!
//! # Relation to `ThreadIndex32`
//!
//! [`crate::thread::ThreadIndex32`] is a 32-bit *witness*: it proves thread
//! uniqueness for a validated 1D launch, and exists only under
//! `#[launch_contract(domain = 1, coordinates = u32)]`. It solves a different
//! problem, and carries a proof this type does not.
//!
//! `Idx32` imposes no launch contract and proves nothing about uniqueness. It
//! is only about the width the arithmetic happens in, and composes with either
//! witness.
//!
//! # Prior art
//!
//! CUDA.jl documents the same concern for Julia's `Int64` promotion in device
//! code. Worth knowing that the question is not unique to Rust - and, on this
//! hardware and toolchain, that the answer is less alarming than it looks.
//!
//! # Overflow
//!
//! Arithmetic wraps in release, like `u32` elsewhere. The `checked_*` forms
//! exist for indices derived from runtime extents, where a product could
//! genuinely exceed `2^32`.

use crate::thread;

/// A 32-bit index, with arithmetic that stays 32-bit.
///
/// Widening happens only in [`Self::get`]. See the module docs for the
/// measurement that motivates it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
#[repr(transparent)]
pub struct Idx32(u32);

impl Idx32 {
    /// Wrap a raw 32-bit index.
    #[must_use]
    #[inline(always)]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The standard 1D thread index, computed entirely in `u32`.
    ///
    /// `blockIdx.x * blockDim.x + threadIdx.x`. Equivalent to
    /// `thread::index_1d()` in value, but the arithmetic never widens.
    ///
    /// This carries no uniqueness proof; it is an index, not a witness. For the
    /// proof, use [`crate::thread::index_1d`] or
    /// [`crate::thread::ThreadIndex32`].
    #[must_use]
    #[inline(always)]
    pub fn thread_1d() -> Self {
        Self(thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x())
    }

    /// Total threads in a 1D launch, in `u32`.
    #[must_use]
    #[inline(always)]
    pub fn grid_extent_1d() -> Self {
        Self(thread::gridDim_x() * thread::blockDim_x())
    }

    /// The index as `usize`, for slice indexing or pointer arithmetic.
    ///
    /// **The single widening point.** Call it as late as possible: once a value
    /// is `usize`, every later operation on it is 64-bit.
    #[must_use]
    #[inline(always)]
    pub const fn get(self) -> usize {
        self.0 as usize
    }

    /// The index as `u32`, without widening.
    #[must_use]
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Multiply by a constant, staying 32-bit.
    ///
    /// The elements-per-thread step: `Idx32::thread_1d().scale(4)`.
    #[must_use]
    #[inline(always)]
    pub const fn scale(self, factor: u32) -> Self {
        Self(self.0.wrapping_mul(factor))
    }

    /// Add a constant, staying 32-bit.
    #[must_use]
    #[inline(always)]
    pub const fn offset(self, delta: u32) -> Self {
        Self(self.0.wrapping_add(delta))
    }

    /// Add another index, staying 32-bit.
    #[must_use]
    #[inline(always)]
    pub const fn add(self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }

    /// `self * factor + delta`, staying 32-bit.
    ///
    /// The whole tile-offset computation in one step, so no intermediate is
    /// tempted into `usize`.
    #[must_use]
    #[inline(always)]
    pub const fn scale_offset(self, factor: u32, delta: u32) -> Self {
        Self(self.0.wrapping_mul(factor).wrapping_add(delta))
    }

    /// Advance by `pass` strides of `stride`, staying 32-bit.
    ///
    /// The grid-stride step.
    #[must_use]
    #[inline(always)]
    pub const fn stride(self, stride: u32, pass: u32) -> Self {
        Self(self.0.wrapping_add(stride.wrapping_mul(pass)))
    }

    /// [`Self::scale`], `None` on overflow.
    #[must_use]
    #[inline(always)]
    pub const fn checked_scale(self, factor: u32) -> Option<Self> {
        match self.0.checked_mul(factor) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// [`Self::offset`], `None` on overflow.
    #[must_use]
    #[inline(always)]
    pub const fn checked_offset(self, delta: u32) -> Option<Self> {
        match self.0.checked_add(delta) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Whether the index is below a bound, compared in 32 bits.
    ///
    /// Using this instead of `idx.get() < len` keeps the comparison narrow. The
    /// bound is a `u32`, so a `usize` length needs one narrowing at the call
    /// site rather than a widening per comparison.
    #[must_use]
    #[inline(always)]
    pub const fn lt(self, bound: u32) -> bool {
        self.0 < bound
    }

    /// Whether the index plus `count` elements fits below a bound.
    ///
    /// The multi-element bound check, done without widening and without
    /// overflowing at the top of the range.
    #[must_use]
    #[inline(always)]
    pub const fn range_fits(self, count: u32, bound: u32) -> bool {
        match self.0.checked_add(count) {
            Some(end) => end <= bound,
            None => false,
        }
    }
}

impl From<u32> for Idx32 {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<Idx32> for u32 {
    #[inline(always)]
    fn from(value: Idx32) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arithmetic must stay in `u32` and match the obvious formula.
    #[test]
    fn arithmetic_matches_the_plain_expression() {
        let i = Idx32::new(7);
        assert_eq!(i.scale(4).raw(), 28);
        assert_eq!(i.offset(3).raw(), 10);
        assert_eq!(i.add(Idx32::new(5)).raw(), 12);
        assert_eq!(i.scale_offset(4, 3).raw(), 31);
        assert_eq!(i.stride(100, 2).raw(), 207);
        assert_eq!(i.get(), 7usize);
    }

    /// `scale_offset` is the fused form, so it must agree with the two-step one.
    #[test]
    fn fused_form_agrees_with_the_steps() {
        for v in [0u32, 1, 7, 1024, 65535] {
            for (f, d) in [(1u32, 0u32), (4, 3), (16, 15), (1024, 1)] {
                assert_eq!(
                    Idx32::new(v).scale_offset(f, d).raw(),
                    Idx32::new(v).scale(f).offset(d).raw(),
                    "disagreement at v={v} f={f} d={d}"
                );
            }
        }
    }

    /// Wrapping is the release behaviour, so it is what the plain methods do;
    /// the checked forms are the opt-in.
    #[test]
    fn overflow_wraps_plainly_and_is_caught_when_checked() {
        let big = Idx32::new(u32::MAX);
        assert_eq!(big.offset(1).raw(), 0, "plain add wraps");
        assert_eq!(big.checked_offset(1), None);
        assert_eq!(big.checked_scale(2), None);
        assert_eq!(Idx32::new(2).checked_scale(3).map(Idx32::raw), Some(6));
        assert_eq!(Idx32::new(2).checked_offset(3).map(Idx32::raw), Some(5));
    }

    /// The multi-element bound check must not overflow into a false pass at the
    /// top of the range - that is the whole reason it is not `i + n <= bound`.
    #[test]
    fn range_check_does_not_wrap_into_a_false_pass() {
        assert!(Idx32::new(0).range_fits(4, 16));
        assert!(Idx32::new(12).range_fits(4, 16), "exact fit at the end");
        assert!(!Idx32::new(13).range_fits(4, 16));
        // Without the checked add, u32::MAX - 1 plus 4 would wrap to 2 and pass.
        assert!(!Idx32::new(u32::MAX - 1).range_fits(4, 16));
        assert!(!Idx32::new(u32::MAX).range_fits(1, u32::MAX));
    }

    #[test]
    fn comparison_is_narrow_and_matches_widened() {
        for v in [0u32, 1, 31, 32, 1000] {
            let i = Idx32::new(v);
            assert_eq!(i.lt(32), i.get() < 32usize);
        }
    }

    /// The type must stay a bare `u32`, since it is meant to be free.
    #[test]
    fn is_a_transparent_u32() {
        assert_eq!(core::mem::size_of::<Idx32>(), core::mem::size_of::<u32>());
        assert_eq!(core::mem::align_of::<Idx32>(), core::mem::align_of::<u32>());
        assert_eq!(u32::from(Idx32::from(42u32)), 42);
    }

    /// Usable in const position, so tile offsets can be folded.
    #[test]
    fn works_at_compile_time() {
        const I: Idx32 = Idx32::new(3).scale_offset(4, 1);
        const AS_USIZE: usize = I.get();
        const _: () = assert!(AS_USIZE == 13);
        assert_eq!(AS_USIZE, 13);
    }
}
