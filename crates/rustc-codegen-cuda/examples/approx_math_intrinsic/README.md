# approx_math_intrinsic

## Safe approximate math intrinsics

This example demonstrates the safe `cuda_device::approx` intrinsics, which map
directly to single-cycle PTX approximate math instructions:

| Function | PTX instruction | Description |
|----------|----------------|-------------|
| `tanh_approx_f32` | `tanh.approx.f32` | Approximate hyperbolic tangent |
| `ex2_approx_ftz_f32` | `ex2.approx.ftz.f32` | Approximate 2^x |
| `rcp_approx_ftz_f32` | `rcp.approx.ftz.f32` | Approximate 1/x |
| `lg2_approx_ftz_f32` | `lg2.approx.ftz.f32` | Approximate log2(x) |

These are **safe** alternatives to writing inline PTX via `ptx_asm!`.

## What This Example Does

A single-thread kernel computes all four approximate math intrinsics on a set
of inputs, writes the results to device memory, and the host verifies each
result against a reference value within hardware tolerance.

Also demonstrates a fast sigmoid composition: `sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5`,
which is a common pattern in neural network activations.

Exits 0 on PASS, 1 on FAIL.

## Run

```bash
cargo oxide run approx_math_intrinsic
```
