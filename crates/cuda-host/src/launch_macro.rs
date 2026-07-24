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
/// # Accepted call shapes
///
/// The launch is captured as an expression, so the module can be reached any
/// way, not just through a bare local. All of these work:
///
/// ```rust,ignore
/// launch!(module.kernel(&stream, cfg, &buf))?;           // local
/// launch!(self.module.kernel(&stream, cfg, &buf))?;      // struct field
/// launch!(self.modules[i].kernel(&stream, cfg, &buf))?;  // indexed
/// launch!((&self.module).kernel(&stream, cfg, &buf))?;   // parenthesized
/// ```
///
/// The struct-field form matters in practice: a forward pass usually keeps its
/// loaded module in `self`, which is exactly the case Gap #5 describes.
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
    // The launch is captured as a single expression rather than as
    // `$module:ident . $kernel:ident (...)`. An `ident` receiver matches only a
    // bare local, so the common case of holding the loaded module in a struct
    // field (`self.module.kernel(..)`) failed to match at all, with a
    // "no rules expected `.`" error. Taking an expression accepts every
    // receiver shape: locals, fields, indexes, and parenthesized expressions.
    ($call:expr) => {
        // SAFETY: Caller is responsible for the kernel launch invariants
        // documented on this macro. Each `launch!` invocation is an
        // independent unsafe boundary.
        unsafe { $call }
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

    /// Verify receivers other than a bare local.
    ///
    /// These are the shapes an `ident` receiver could not match: matching on
    /// `$module:ident . $kernel:ident (..)` rejected `host.module.kern(..)`
    /// outright with "no rules expected `.`". A forward pass normally keeps its
    /// loaded module in a struct field, so this is the common case, not an edge
    /// case.
    #[test]
    fn test_launch_macro_accepts_non_ident_receivers() {
        struct FakeModule;

        impl FakeModule {
            unsafe fn kern(&self, a: i32) -> Result<i32, String> {
                Ok(a * 2)
            }
        }

        struct Host {
            module: FakeModule,
            modules: Vec<FakeModule>,
        }

        let host = Host {
            module: FakeModule,
            modules: vec![FakeModule],
        };

        // Struct field receiver.
        assert_eq!(launch!(host.module.kern(21)), Ok(42));
        // Indexed receiver.
        assert_eq!(launch!(host.modules[0].kern(50)), Ok(100));
        // Parenthesized receiver.
        assert_eq!(launch!((&host.module).kern(5)), Ok(10));
    }
}
