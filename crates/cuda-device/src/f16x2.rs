// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `f16x2` arithmetic intrinsics.
//!
//! Each `u32` stores two f16 values. The first value uses the low 16 bits.
//! The second value uses the high 16 bits.
//!
//! # When the packed path is worth reaching for
//!
//! Scalar `f16` arithmetic does not beat `f32`. Measured on sm_86, the same
//! fused multiply-add over two elements, written three ways:
//!
//! ```text
//! kernel          SASS   half-ops        loads
//! f16x2 packed      24   1x HFMA2        2x LDG.E.CONSTANT
//! f16 scalar        32   2x HFMA2        4x LDG.E.U16
//! f32 scalar        32   -               4x LDG.E
//! ```
//!
//! Scalar f16 ties f32 exactly, which is the result worth knowing before
//! reaching for `f16` expecting a speedup.
//!
//! The reason is not what it looks like. `ptxas` *does* vectorize the
//! arithmetic - `HFMA2` appears in the scalar kernel too, it simply issues one
//! per element and uses half the lanes. What the scalar form loses is the
//! memory access: four 16-bit `LDG.E.U16` where the packed form does two 32-bit
//! loads. The packed win is mostly load width, not math throughput.
//!
//! That places it under the same rule as every other element type: transaction
//! width follows the alignment of the type you index, and a `&[f16]` is
//! 2-byte aligned. Reading pairs as `u32` is what makes the loads wide, and the
//! packed arithmetic then comes along for free because the operands are already
//! in the right register layout.
//!
//! So the guidance is narrower than "use f16 for speed":
//!
//! - Moving or staging f16 data: read it as `u32` pairs (or wider) and unpack
//!   with [`crate::convert::cvt_f32x2_f16x2`]. The win is the load.
//! - Elementwise f16 math on data already in registers: use the packed ops
//!   here, which is the layout the loads already produced.
//! - Scalar `f16` in a loop: expect `f32` performance, not better. If the
//!   working set is what matters, the saving is in bandwidth and storage, not
//!   in the arithmetic.
//!
//! Nothing converts a scalar loop to the packed form automatically; the packed
//! path is opt-in, and these intrinsics are how it is spelled.

include!("generated/f16x2.rs");
