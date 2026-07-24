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
//! # When to use
//!
//! Use these helpers when a warp reads or writes contiguous, naturally-aligned
//! data and you want to guarantee that the compiler emits the widest possible
//! memory transaction instead of falling back to multiple scalar loads. Common
//! use cases include:
//!
//! - Copying tiles between global and shared memory
//! - Streaming reductions over large arrays
//! - Any bandwidth-bound kernel where coalesced wide loads help

// ---------------------------------------------------------------------------
// f32 vectorized loads
// ---------------------------------------------------------------------------

/// Load two consecutive `f32` values with a single 64-bit global memory
/// transaction.
///
/// Emits `ld.global.v2.f32 {%0, %1}, [%2];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 2 contiguous `f32` values.
/// - `ptr` must be aligned to 8 bytes (natural alignment for a 64-bit load).
/// - The memory region must be readable by the calling thread.
#[inline(always)]
pub unsafe fn load_v2_f32(ptr: *const f32) -> (f32, f32) {
    let v0: f32;
    let v1: f32;
    unsafe {
        crate::ptx_asm!(
            "ld.global.v2.f32 {%0, %1}, [%2];",
            out("=f") v0,
            out("=f") v1,
            in("l") ptr,
        );
    }
    (v0, v1)
}

/// Load four consecutive `f32` values with a single 128-bit global memory
/// transaction.
///
/// Emits `ld.global.v4.f32 {%0, %1, %2, %3}, [%4];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 4 contiguous `f32` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit load).
/// - The memory region must be readable by the calling thread.
#[inline(always)]
pub unsafe fn load_v4_f32(ptr: *const f32) -> (f32, f32, f32, f32) {
    let v0: f32;
    let v1: f32;
    let v2: f32;
    let v3: f32;
    unsafe {
        crate::ptx_asm!(
            "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];",
            out("=f") v0,
            out("=f") v1,
            out("=f") v2,
            out("=f") v3,
            in("l") ptr,
        );
    }
    (v0, v1, v2, v3)
}

// ---------------------------------------------------------------------------
// f32 vectorized stores
// ---------------------------------------------------------------------------

/// Store two `f32` values with a single 64-bit global memory transaction.
///
/// Emits `st.global.v2.f32 [%0], {%1, %2};`
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
            "st.global.v2.f32 [%0], {%1, %2};",
            in("l") ptr,
            in("f") v0,
            in("f") v1,
        );
    }
}

/// Store four `f32` values with a single 128-bit global memory transaction.
///
/// Emits `st.global.v4.f32 [%0], {%1, %2, %3, %4};`
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
            "st.global.v4.f32 [%0], {%1, %2, %3, %4};",
            in("l") ptr,
            in("f") v0,
            in("f") v1,
            in("f") v2,
            in("f") v3,
        );
    }
}

// ---------------------------------------------------------------------------
// u32 vectorized loads
// ---------------------------------------------------------------------------

/// Load two consecutive `u32` values with a single 64-bit global memory
/// transaction.
///
/// Emits `ld.global.v2.u32 {%0, %1}, [%2];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 2 contiguous `u32` values.
/// - `ptr` must be aligned to 8 bytes (natural alignment for a 64-bit load).
/// - The memory region must be readable by the calling thread.
#[inline(always)]
pub unsafe fn load_v2_u32(ptr: *const u32) -> (u32, u32) {
    let v0: u32;
    let v1: u32;
    unsafe {
        crate::ptx_asm!(
            "ld.global.v2.u32 {%0, %1}, [%2];",
            out("=r") v0,
            out("=r") v1,
            in("l") ptr,
        );
    }
    (v0, v1)
}

/// Load four consecutive `u32` values with a single 128-bit global memory
/// transaction.
///
/// Emits `ld.global.v4.u32 {%0, %1, %2, %3}, [%4];`
///
/// # Safety
///
/// - `ptr` must be a valid device pointer to at least 4 contiguous `u32` values.
/// - `ptr` must be aligned to 16 bytes (natural alignment for a 128-bit load).
/// - The memory region must be readable by the calling thread.
#[inline(always)]
pub unsafe fn load_v4_u32(ptr: *const u32) -> (u32, u32, u32, u32) {
    let v0: u32;
    let v1: u32;
    let v2: u32;
    let v3: u32;
    unsafe {
        crate::ptx_asm!(
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") v0,
            out("=r") v1,
            out("=r") v2,
            out("=r") v3,
            in("l") ptr,
        );
    }
    (v0, v1, v2, v3)
}

// ---------------------------------------------------------------------------
// u32 vectorized stores
// ---------------------------------------------------------------------------

/// Store two `u32` values with a single 64-bit global memory transaction.
///
/// Emits `st.global.v2.u32 [%0], {%1, %2};`
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
            "st.global.v2.u32 [%0], {%1, %2};",
            in("l") ptr,
            in("r") v0,
            in("r") v1,
        );
    }
}

/// Store four `u32` values with a single 128-bit global memory transaction.
///
/// Emits `st.global.v4.u32 [%0], {%1, %2, %3, %4};`
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
            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
            in("l") ptr,
            in("r") v0,
            in("r") v1,
            in("r") v2,
            in("r") v3,
        );
    }
}
