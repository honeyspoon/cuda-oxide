/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block-level collective primitives.
//!
//! These utilities combine warp-level shuffles with shared memory and
//! `sync_threads()` to perform reductions across an entire thread block.
//!
//! # Requirements
//!
//! - The caller must declare a `static mut SharedArray<f32, N>` where
//!   `N >= block_size / 32` (one slot per warp).
//! - All threads in the block must participate (no early-exit before the call).

use crate::thread;
use crate::warp;

/// Reduce-sum a scalar f32 across all threads in the block.
///
/// Returns the sum in **lane 0 of warp 0 only**. All other threads get
/// an unspecified value.
///
/// # Arguments
///
/// * `val` - Each thread's contribution to the sum
/// * `smem` - Pointer to shared memory with at least `block_size / 32` f32 slots
/// * `block_size` - Total number of threads in the block (must be a multiple of 32)
///
/// # Example
///
/// ```rust,ignore
/// static mut SMEM: SharedArray<f32, 32> = SharedArray::UNINIT;
///
/// let sp = unsafe { core::ptr::addr_of_mut!(SMEM) as *mut f32 };
/// let total = block::reduce_sum_f32(my_value, sp, 256);
/// if thread::threadIdx_x() == 0 {
///     // `total` is the sum of all 256 threads' values
/// }
/// ```
#[inline(always)]
pub fn reduce_sum_f32(val: f32, smem: *mut f32, block_size: u32) -> f32 {
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_id = tid / 32;
    let num_warps = block_size / 32;

    // Step 1: intra-warp reduce via butterfly shuffles
    let warp_sum = warp::reduce_sum_f32(val);

    // Step 2: lane 0 of each warp writes to shared memory
    if lane == 0 {
        unsafe { smem.add(warp_id as usize).write(warp_sum) };
    }
    thread::sync_threads();

    // Step 3: warp 0 reduces across all warps
    let result = if warp_id == 0 {
        let v = if tid < num_warps {
            unsafe { smem.add(tid as usize).read() }
        } else {
            0.0
        };
        warp::reduce_sum_f32(v)
    } else {
        0.0
    };

    result
}

/// Reduce-max a scalar f32 across all threads in the block.
///
/// Returns the maximum in **lane 0 of warp 0 only**. All other threads
/// get an unspecified value.
///
/// # Arguments
///
/// * `val` - Each thread's value
/// * `smem` - Pointer to shared memory with at least `block_size / 32` f32 slots
/// * `block_size` - Total number of threads in the block (must be a multiple of 32)
///
/// # Example
///
/// ```rust,ignore
/// static mut SMEM: SharedArray<f32, 32> = SharedArray::UNINIT;
///
/// let sp = unsafe { core::ptr::addr_of_mut!(SMEM) as *mut f32 };
/// let block_max = block::reduce_max_f32(my_value, sp, 256);
/// if thread::threadIdx_x() == 0 {
///     // `block_max` is the maximum across all 256 threads
/// }
/// ```
#[inline(always)]
pub fn reduce_max_f32(val: f32, smem: *mut f32, block_size: u32) -> f32 {
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_id = tid / 32;
    let num_warps = block_size / 32;

    // Step 1: intra-warp reduce via butterfly shuffles
    let warp_max = warp::reduce_max_f32(val);

    // Step 2: lane 0 of each warp writes to shared memory
    if lane == 0 {
        unsafe { smem.add(warp_id as usize).write(warp_max) };
    }
    thread::sync_threads();

    // Step 3: warp 0 reduces across all warps
    let result = if warp_id == 0 {
        let v = if tid < num_warps {
            unsafe { smem.add(tid as usize).read() }
        } else {
            f32::NEG_INFINITY
        };
        warp::reduce_max_f32(v)
    } else {
        f32::NEG_INFINITY
    };

    result
}
