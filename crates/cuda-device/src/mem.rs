/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vectorized global memory load/store intrinsics.
//!
//! These functions emit wide memory transactions (64-bit or 128-bit) that
//! improve memory bandwidth utilization compared to scalar 32-bit loads.
//!
//! # Alignment
//!
//! All pointers must be naturally aligned to the transaction width:
//! - `v2` operations require 8-byte alignment
//! - `v4` operations require 16-byte alignment
//!
//! Misaligned access causes a hardware exception.
//!
//! # Prefer alignment-driven vectorization
//!
//! These helpers are the raw-pointer escape hatch, not the first choice. The
//! compiler already fuses a whole-element copy into `ld.global.v4` /
//! `st.global.v4` when the element type carries the matching alignment, with
//! no `unsafe` and no inline PTX:
//!
//! ```rust,ignore
//! #[repr(C, align(16))]
//! #[derive(Clone, Copy)]
//! pub struct F32x4([f32; 4]);
//!
//! // Compiles to a single 128-bit load plus a 128-bit store.
//! #[kernel]
//! pub fn copy(input: &[F32x4], mut output: DisjointSlice<F32x4>) {
//!     let idx = thread::index_1d();
//!     if let Some(o) = output.get_mut(idx) {
//!         *o = input[idx.get()];
//!     }
//! }
//! ```
//!
//! See the `vectorization` example for the full alignment-to-width table.
//! Reach for the functions below only when the address is a raw pointer whose
//! alignment the type system cannot express, such as a manually computed tile
//! offset into a larger buffer.
//!
//! # Address space
//!
//! A `.global`-qualified PTX access requires an address in the global window,
//! but Rust raw pointers reach inline asm as *generic* (address space 0)
//! pointers. Each template therefore converts with `cvta.to.global.u64` before
//! the access; passing a generic pointer straight to `ld.global` is not
//! guaranteed to address the intended memory. The conversion is confined to a
//! braced block so the `.reg` declaration does not collide when several of
//! these helpers are inlined into one kernel.
//!
//! # Ordering
//!
//! Every template carries `clobber("memory")`. Without it the compiler is free
//! to move ordinary loads and stores across the asm, since inline asm is
//! assumed not to touch memory unless it says otherwise. The clobber is what
//! makes a store here visible to a later read of the same address.

// ---------------------------------------------------------------------------
// f32 vectorized loads
// ---------------------------------------------------------------------------

/// Load two consecutive `f32` values with a single 64-bit global memory
/// transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `ld.global.v2.f32 {v0, v1}, [ptr];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 2 contiguous `f32` values.
/// - `ptr` must be aligned to 8 bytes (natural alignment for a 64-bit load).
/// - The memory region must be readable by the calling thread.
#[must_use]
#[inline(always)]
pub unsafe fn load_v2_f32(ptr: *const f32) -> (f32, f32) {
    let v0: f32;
    let v1: f32;
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %2; ld.global.v2.f32 {%0, %1}, [%%gmem64]; }",
            out("=f") v0,
            out("=f") v1,
            in("l") ptr,
            clobber("memory"),
        );
    }
    (v0, v1)
}

/// Load four consecutive `f32` values with a single 128-bit global memory
/// transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `ld.global.v4.f32 {v0, v1, v2, v3}, [ptr];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 4 contiguous `f32` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit load).
/// - The memory region must be readable by the calling thread.
#[must_use]
#[inline(always)]
pub unsafe fn load_v4_f32(ptr: *const f32) -> (f32, f32, f32, f32) {
    let v0: f32;
    let v1: f32;
    let v2: f32;
    let v3: f32;
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %4; ld.global.v4.f32 {%0, %1, %2, %3}, [%%gmem64]; }",
            out("=f") v0,
            out("=f") v1,
            out("=f") v2,
            out("=f") v3,
            in("l") ptr,
            clobber("memory"),
        );
    }
    (v0, v1, v2, v3)
}

// ---------------------------------------------------------------------------
// f32 vectorized stores
// ---------------------------------------------------------------------------

/// Store two `f32` values with a single 64-bit global memory transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `st.global.v2.f32 [ptr], {v0, v1};`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer with room for at least 2 contiguous
///   `f32` values.
/// - `ptr` must be aligned to 8 bytes (natural alignment for a 64-bit store).
/// - The memory region must be writable by the calling thread.
/// - No other thread may concurrently read or write the same memory without
///   explicit synchronization.
#[inline(always)]
pub unsafe fn store_v2_f32(ptr: *mut f32, v0: f32, v1: f32) {
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %0; st.global.v2.f32 [%%gmem64], {%1, %2}; }",
            in("l") ptr,
            in("f") v0,
            in("f") v1,
            clobber("memory"),
        );
    }
}

/// Store four `f32` values with a single 128-bit global memory transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `st.global.v4.f32 [ptr], {v0, v1, v2, v3};`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer with room for at least 4 contiguous
///   `f32` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit store).
/// - The memory region must be writable by the calling thread.
/// - No other thread may concurrently read or write the same memory without
///   explicit synchronization.
#[inline(always)]
pub unsafe fn store_v4_f32(ptr: *mut f32, v0: f32, v1: f32, v2: f32, v3: f32) {
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %0; st.global.v4.f32 [%%gmem64], {%1, %2, %3, %4}; }",
            in("l") ptr,
            in("f") v0,
            in("f") v1,
            in("f") v2,
            in("f") v3,
            clobber("memory"),
        );
    }
}

// ---------------------------------------------------------------------------
// u32 vectorized loads
// ---------------------------------------------------------------------------

/// Load two consecutive `u32` values with a single 64-bit global memory
/// transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `ld.global.v2.u32 {v0, v1}, [ptr];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 2 contiguous `u32` values.
/// - `ptr` must be aligned to 8 bytes (natural alignment for a 64-bit load).
/// - The memory region must be readable by the calling thread.
#[must_use]
#[inline(always)]
pub unsafe fn load_v2_u32(ptr: *const u32) -> (u32, u32) {
    let v0: u32;
    let v1: u32;
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %2; ld.global.v2.u32 {%0, %1}, [%%gmem64]; }",
            out("=r") v0,
            out("=r") v1,
            in("l") ptr,
            clobber("memory"),
        );
    }
    (v0, v1)
}

/// Load four consecutive `u32` values with a single 128-bit global memory
/// transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `ld.global.v4.u32 {v0, v1, v2, v3}, [ptr];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 4 contiguous `u32` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit load).
/// - The memory region must be readable by the calling thread.
#[must_use]
#[inline(always)]
pub unsafe fn load_v4_u32(ptr: *const u32) -> (u32, u32, u32, u32) {
    let v0: u32;
    let v1: u32;
    let v2: u32;
    let v3: u32;
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %4; ld.global.v4.u32 {%0, %1, %2, %3}, [%%gmem64]; }",
            out("=r") v0,
            out("=r") v1,
            out("=r") v2,
            out("=r") v3,
            in("l") ptr,
            clobber("memory"),
        );
    }
    (v0, v1, v2, v3)
}

// ---------------------------------------------------------------------------
// u32 vectorized stores
// ---------------------------------------------------------------------------

/// Store two `u32` values with a single 64-bit global memory transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `st.global.v2.u32 [ptr], {v0, v1};`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer with room for at least 2 contiguous
///   `u32` values.
/// - `ptr` must be aligned to 8 bytes (natural alignment for a 64-bit store).
/// - The memory region must be writable by the calling thread.
/// - No other thread may concurrently read or write the same memory without
///   explicit synchronization.
#[inline(always)]
pub unsafe fn store_v2_u32(ptr: *mut u32, v0: u32, v1: u32) {
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %0; st.global.v2.u32 [%%gmem64], {%1, %2}; }",
            in("l") ptr,
            in("r") v0,
            in("r") v1,
            clobber("memory"),
        );
    }
}

/// Store four `u32` values with a single 128-bit global memory transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `st.global.v4.u32 [ptr], {v0, v1, v2, v3};`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer with room for at least 4 contiguous
///   `u32` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit store).
/// - The memory region must be writable by the calling thread.
/// - No other thread may concurrently read or write the same memory without
///   explicit synchronization.
#[inline(always)]
pub unsafe fn store_v4_u32(ptr: *mut u32, v0: u32, v1: u32, v2: u32, v3: u32) {
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %0; st.global.v4.u32 [%%gmem64], {%1, %2, %3, %4}; }",
            in("l") ptr,
            in("r") v0,
            in("r") v1,
            in("r") v2,
            in("r") v3,
            clobber("memory"),
        );
    }
}

// ---------------------------------------------------------------------------
// f64 vectorized loads
// ---------------------------------------------------------------------------

/// Load two consecutive `f64` values with a single 128-bit global memory
/// transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `ld.global.v2.f64 {v0, v1}, [ptr];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 2 contiguous `f64` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit load).
/// - The memory region must be readable by the calling thread.
#[must_use]
#[inline(always)]
pub unsafe fn load_v2_f64(ptr: *const f64) -> (f64, f64) {
    let v0: f64;
    let v1: f64;
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %2; ld.global.v2.f64 {%0, %1}, [%%gmem64]; }",
            out("=d") v0,
            out("=d") v1,
            in("l") ptr,
            clobber("memory"),
        );
    }
    (v0, v1)
}

// ---------------------------------------------------------------------------
// f64 vectorized stores
// ---------------------------------------------------------------------------

/// Store two `f64` values with a single 128-bit global memory transaction.
///
/// Emits `cvta.to.global.u64` to move `ptr` into the global window,
/// then `st.global.v2.f64 [ptr], {v0, v1};`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer with room for at least 2 contiguous
///   `f64` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit store).
/// - The memory region must be writable by the calling thread.
/// - No other thread may concurrently read or write the same memory without
///   explicit synchronization.
#[inline(always)]
pub unsafe fn store_v2_f64(ptr: *mut f64, v0: f64, v1: f64) {
    unsafe {
        crate::ptx_asm!(
            "{ .reg .u64 %%gmem64; cvta.to.global.u64 %%gmem64, %0; st.global.v2.f64 [%%gmem64], {%1, %2}; }",
            in("l") ptr,
            in("d") v0,
            in("d") v1,
            clobber("memory"),
        );
    }
}
