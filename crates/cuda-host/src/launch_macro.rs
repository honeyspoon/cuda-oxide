/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Convenience macro for isolating individual kernel launch `unsafe` blocks.
//!
//! See [`launch!`] for full documentation.

/// Wraps a single kernel launch, isolating the `unsafe` to one call site.
///
/// # Motivation
///
/// Without this macro, forward pass code looks like:
///
/// ```rust,ignore
/// unsafe {
///     // 100+ kernel launches in one giant unsafe block
///     module.kernel_a(&stream, cfg, &buf_a, &mut out_a)?;
///     module.kernel_b(&stream, cfg, &buf_b, &mut out_b)?;
///     // ...
/// }
/// ```
///
/// With this macro, each launch is self-contained:
///
/// ```rust,ignore
/// launch!(module.kernel_a(&stream, cfg, &buf_a, &mut out_a))?;
/// launch!(module.kernel_b(&stream, cfg, &buf_b, &mut out_b))?;
/// ```
///
/// This doesn't change the safety semantics (kernel launches are still
/// inherently unsafe due to GPU memory access patterns), but it makes
/// the unsafe boundary explicit at each launch site rather than one
/// all-encompassing block.
///
/// # Difference from `cuda_launch!`
///
/// [`cuda_launch!`] is a low-level macro that marshals raw kernel names and
/// argument lists into `cuLaunchKernel` calls. It operates at the driver API
/// level and requires the caller to manage `Vec<*mut c_void>` argument arrays.
///
/// `launch!` is a high-level convenience wrapper for the typed, generated
/// launch methods that `#[cuda_module]` produces. It simply wraps
/// `module.kernel(...)` in an `unsafe` block so that each launch site is
/// individually annotated rather than bundled into one giant `unsafe` region.
///
/// # Safety
///
/// The macro expands to an `unsafe` block around the kernel call.
/// The caller is responsible for ensuring:
/// - All buffer arguments are valid for the kernel's access pattern
/// - The launch configuration matches the kernel's requirements
/// - Output buffers have sufficient size
///
/// Each invocation is an independent unsafe boundary, making it easier to
/// audit which invariants apply to which kernel launch.
///
/// This is a stopgap until upstream proof-carrying device views
/// (branch `feat/proof-carrying-views`) and typed launch contracts
/// (branch `feat/typed-launch-contracts`) provide compile-time safety.
///
/// # Examples
///
/// Single kernel launch:
///
/// ```rust,ignore
/// use cuda_host::launch;
///
/// let result = launch!(module.vecadd(&stream, config, &a_dev, &b_dev, &mut c_dev))?;
/// ```
///
/// Multiple independent launches with individual safety boundaries:
///
/// ```rust,ignore
/// use cuda_host::launch;
///
/// // Each launch is its own unsafe boundary — reviewers can verify
/// // buffer validity for each kernel independently.
/// launch!(module.embed(&stream, cfg, &token_ids, &mut embeddings))?;
/// launch!(module.layer_norm(&stream, cfg, &embeddings, &weights, &mut normed))?;
/// launch!(module.attention(&stream, cfg, &normed, &kv_cache, &mut attn_out))?;
/// ```
#[macro_export]
macro_rules! launch {
    ($module:ident . $kernel:ident ( $($args:expr),* $(,)? )) => {
        // SAFETY: Caller is responsible for the kernel launch invariants
        // documented on this macro. Each `launch!` invocation is an
        // independent unsafe boundary.
        unsafe { $module.$kernel( $($args),* ) }
    };
}

#[cfg(test)]
mod tests {
    /// Verify that `launch!` compiles for a method call that returns `Result`.
    ///
    /// We can't test actual kernel launches without a GPU context, but we can
    /// ensure the macro expansion is syntactically correct and propagates
    /// return values.
    #[test]
    fn test_launch_macro_expansion() {
        struct FakeModule;

        impl FakeModule {
            /// Simulates a kernel launch method signature.
            ///
            /// # Safety
            /// This is a test double; no actual device launch occurs.
            unsafe fn test_kernel(&self, _x: i32, _y: &[f32]) -> Result<(), String> {
                Ok(())
            }
        }

        let module = FakeModule;
        let data = vec![1.0f32, 2.0, 3.0];

        let result = launch!(module.test_kernel(42, &data));
        assert!(result.is_ok());
    }

    /// Verify that `launch!` works with the `?` operator for error propagation.
    #[test]
    fn test_launch_macro_with_question_mark() -> Result<(), String> {
        struct FakeModule;

        impl FakeModule {
            unsafe fn my_kernel(&self, _a: &[u8], _b: &mut [u8]) -> Result<(), String> {
                Ok(())
            }
        }

        let module = FakeModule;
        let input = vec![1u8, 2, 3];
        let mut output = vec![0u8; 3];

        launch!(module.my_kernel(&input, &mut output))?;
        Ok(())
    }

    /// Verify that the macro accepts trailing commas.
    #[test]
    fn test_launch_macro_trailing_comma() {
        struct FakeModule;

        impl FakeModule {
            unsafe fn kern(&self, _a: i32, _b: i32) -> Result<(), String> {
                Ok(())
            }
        }

        let module = FakeModule;
        let result = launch!(module.kern(1, 2,));
        assert!(result.is_ok());
    }
}
