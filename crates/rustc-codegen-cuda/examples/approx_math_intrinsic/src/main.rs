// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe approximate math intrinsics — `tanh`, `ex2`, `rcp`, `lg2`.
//!
//! Demonstrates `cuda_device::approx::*` as safe replacements for inline PTX.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::approx::{
    ex2_approx_ftz_f32, lg2_approx_ftz_f32, rcp_approx_ftz_f32, tanh_approx_f32,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Number of result slots per test point.
const NUM_OPS: usize = 5;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test_approx_math(input: &[f32], mut out: DisjointSlice<[f32; NUM_OPS]>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(row) = out.get_mut(idx) {
            let x = input[idx_raw];

            row[0] = tanh_approx_f32(x);
            row[1] = ex2_approx_ftz_f32(x);
            row[2] = rcp_approx_ftz_f32(x);
            row[3] = lg2_approx_ftz_f32(x);

            // Fast sigmoid: 0.5 * tanh(0.5 * x) + 0.5
            row[4] = 0.5 * tanh_approx_f32(0.5 * x) + 0.5;
        }
    }
}

fn approx_eq(got: f32, expected: f32, rel_tol: f32) -> bool {
    if expected == 0.0 {
        got.abs() < rel_tol
    } else {
        ((got - expected) / expected).abs() < rel_tol
    }
}

fn check(label: &str, got: f32, expected: f32, tol: f32) -> bool {
    let ok = approx_eq(got, expected, tol);
    if ok {
        println!("  {label}: ok  (got={got:.6}, expected={expected:.6})");
    } else {
        println!("  {label}: FAIL  (got={got:.6}, expected={expected:.6})");
    }
    ok
}

fn main() {
    println!("=== approx_math_intrinsic ===");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");

    let inputs: Vec<f32> = vec![1.0, -1.0, 0.5, 2.0];
    let n = inputs.len();

    let d_input = DeviceBuffer::from_host(&stream, &inputs).unwrap();
    let mut d_out = DeviceBuffer::<[f32; NUM_OPS]>::zeroed(&stream, n).unwrap();

    // SAFETY: launch shape matches kernel; buffers cover all accesses.
    unsafe {
        module.test_approx_math(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            &d_input,
            &mut d_out,
        )
    }
    .expect("launch test_approx_math");

    let rows = d_out.to_host_vec(&stream).unwrap();

    // Hardware approximate instructions have limited precision:
    // - tanh.approx: ~1e-3 relative error
    // - ex2.approx:  ~1e-4 relative error
    // - rcp.approx:  ~1e-3 relative error
    // - lg2.approx:  ~1e-4 relative error
    let tol = 0.01; // 1% relative tolerance covers all four

    let mut pass = true;
    for (i, x) in inputs.iter().enumerate() {
        println!("x = {x}:");
        let row = &rows[i];

        pass &= check("tanh", row[0], x.tanh(), tol);
        pass &= check("ex2", row[1], (2.0_f32).powf(*x), tol);
        if *x != 0.0 {
            pass &= check("rcp", row[2], 1.0 / x, tol);
        }
        if *x > 0.0 {
            pass &= check("lg2", row[3], x.log2(), tol);
        }

        let sigmoid_ref = 1.0 / (1.0 + (-x).exp());
        pass &= check("sigmoid", row[4], sigmoid_ref, tol);
    }

    if !pass {
        println!("FAIL: approx_math_intrinsic");
        std::process::exit(1);
    }
    println!("PASS: approx_math_intrinsic, all approximate math intrinsics verified");
}
