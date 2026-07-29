/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA graph capture and replay (RAII), **exploratory**.
//!
//! This module exists to find out what a safe graph API can and cannot promise.
//! It wraps the smallest useful slice of the graph surface — stream capture,
//! instantiation, replay — and records, in prose next to each item, which
//! obligations the type system discharges and which it merely documents.
//!
//! # Why bother
//!
//! Graph replay is worth a fixed ~1 µs per launch. Measured on this repo's
//! benchmark harness (RTX A2000, sm_86, 500 iterations):
//!
//! ```text
//! path    direct P50   graph P50   saved    nodes   µs/node
//! FP16    3.867 ms     3.697 ms    0.170    173     0.98
//! INT8    4.670 ms     4.491 ms    0.179    222     0.81
//! W8A16   9.180 ms     9.002 ms    0.178    293     0.61
//! ```
//!
//! The saving is near-constant while node counts differ by 70%, which is what
//! identifies it as per-launch overhead rather than anything kernel-related.
//!
//! # What the types actually enforce
//!
//! **Capture must be terminated even when it fails.** This is the obligation
//! most easily missed and the main reason to prefer a guard. The driver
//! documents that after an error, the stream enters
//! `CU_STREAM_CAPTURE_STATUS_INVALIDATED` and
//!
//! > The capture sequence must be terminated with `cuStreamEndCapture` on the
//! > stream where it was initiated in order to continue using `hStream`.
//!
//! So an early return or panic between begin and end leaves the stream
//! permanently unusable — and, because a capturing blocking stream also puts
//! the legacy null stream in "an unusable state", it can take the default
//! stream down with it. [`StreamCapture`] terminates in [`Drop`], so unwinding
//! cannot poison the stream.
//!
//! **Replays of one executable graph are serialised.** [`GraphExec::launch`]
//! takes `&mut self`. Two concurrent replays of the same `CUgraphExec` are not
//! permitted, and `&mut` is exactly that constraint, checked at compile time
//! rather than documented.
//!
//! # What the types do NOT enforce — read before relying on this
//!
//! **Captured graphs freeze device pointers, and nothing here ties their
//! lifetimes together.** Capture records the *argument values* that were
//! enqueued, including raw device addresses. If a buffer used during capture is
//! dropped while a [`GraphExec`] built from that capture is still alive, replay
//! dereferences freed device memory. There is no borrow relating the two:
//!
//! ```ignore
//! let exec = {
//!     let scratch = DeviceBuffer::<f32>::new(&ctx, n)?;   // dropped here
//!     let capture = stream.begin_capture(CaptureMode::Global)?;
//!     launch_using(&scratch)?;
//!     capture.end()?.instantiate()?
//! };
//! exec.launch(&stream)?;   // use-after-free; compiles today
//! ```
//!
//! **Measured, not assumed** (`tests/graph_capture.rs`, the `--ignored` probe, on
//! an RTX A2000). The driver does not diagnose this, and the result looks right:
//!
//! ```text
//! launch    = Ok(())
//! sync      = Ok(())
//! read      = Ok(())
//! dst       = [7, 8, 9, 10]   (captured source was [7, 8, 9, 10])
//! diagnosed = false
//! ```
//!
//! The replay read freed device memory that still happened to hold the old
//! bytes, so the copy "succeeded". That is the worst available failure mode: it
//! passes in testing and corrupts once the freed pages are reused, with no error
//! at launch, at synchronize, or on read-back. Stream-ordered deallocation
//! returning memory to a pool rather than the driver makes this the *likely*
//! presentation, not a rare one.
//!
//! Expressing the constraint would require the exec to borrow every buffer
//! touched during capture, which the capture API cannot observe — the driver
//! sees pointer values, not Rust borrows. Options, none free: thread a lifetime
//! through a capture-scope closure that owns the buffers, or make graphs own
//! `Arc` clones of their inputs.
//!
//! **[`CapturedGraph::instantiate`] is therefore `unsafe`**, carrying this as its
//! documented obligation. That is the honest signature given the failure is
//! silent: a safe one would promise a check nothing performs. Everything else
//! here is safe, so the `unsafe` marks exactly the one step that takes on the
//! lifetime responsibility.
//!
//! Also not modelled, deliberately: graphs containing memory allocation nodes
//! (the driver permits at most one live exec per graph in that case), device
//! launch, and node mutation. Mutation in particular interacts with launch
//! contracts — see the note on [`GraphExec`].

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use core::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// How a capture treats concurrent activity in other threads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaptureMode {
    /// Prohibit potentially unsafe driver actions in *any* thread for the
    /// duration of the capture. The conservative default.
    #[default]
    Global,
    /// Prohibit them only in the capturing thread.
    ThreadLocal,
    /// Prohibit nothing; the caller guarantees no interfering action occurs.
    Relaxed,
}

impl CaptureMode {
    fn as_raw(self) -> cuda_bindings::CUstreamCaptureMode {
        match self {
            Self::Global => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => {
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            }
            Self::Relaxed => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

/// An in-progress stream capture.
///
/// Work enqueued on the stream while this guard is alive is recorded into a
/// graph instead of executing. Call [`end`](Self::end) to finish and obtain the
/// [`CapturedGraph`].
///
/// Dropping the guard without calling `end` terminates the capture and discards
/// the graph. That is not merely tidy: an unterminated capture leaves the stream
/// unusable, so termination has to happen on the unwind path too.
#[must_use = "dropping the guard immediately ends the capture and discards the graph"]
pub struct StreamCapture<'stream> {
    stream: &'stream CudaStream,
    /// Cleared by `end` so `Drop` does not terminate the capture twice.
    active: bool,
}

impl<'stream> StreamCapture<'stream> {
    /// Finishes the capture and returns the recorded graph.
    ///
    /// Returns an error if the capture was invalidated. Note that ending an
    /// invalidated capture is *required* to make the stream usable again, so
    /// the error path here has already done the necessary cleanup.
    pub fn end(mut self) -> Result<CapturedGraph, DriverError> {
        self.active = false;
        self.stream.context().bind_to_thread()?;
        let mut graph = MaybeUninit::<cuda_bindings::CUgraph>::uninit();
        let code = unsafe {
            cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), graph.as_mut_ptr())
        };
        let cu_graph = (code, graph).result()?;
        if cu_graph.is_null() {
            // `cuStreamEndCapture` reports success with a null graph when the
            // capture was invalidated by an earlier error, so a bare status
            // check is not enough to conclude a graph exists.
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        Ok(CapturedGraph {
            cu_graph,
            ctx: Arc::clone(self.stream.context()),
        })
    }
}

impl Drop for StreamCapture<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Terminate and discard. Errors are unreportable here, but the call
        // must still happen: skipping it leaves the stream unusable, which is a
        // worse outcome than a leaked graph handle.
        let mut graph = MaybeUninit::<cuda_bindings::CUgraph>::uninit();
        unsafe {
            let _ = cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), graph.as_mut_ptr());
            let handle = graph.assume_init();
            if !handle.is_null() {
                let _ = cuda_bindings::cuGraphDestroy(handle);
            }
        }
    }
}

/// A captured, not-yet-executable graph (`CUgraph`).
#[derive(Debug)]
pub struct CapturedGraph {
    cu_graph: cuda_bindings::CUgraph,
    ctx: Arc<CudaContext>,
}

impl CapturedGraph {
    /// Number of nodes recorded, useful for confirming a capture saw the work
    /// it was meant to see.
    pub fn node_count(&self) -> Result<usize, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut count: usize = 0;
        unsafe {
            cuda_bindings::cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut count)
                .result()?;
        }
        Ok(count)
    }

    /// Instantiates an executable graph.
    ///
    /// # Safety
    ///
    /// Every device allocation referenced by the captured work must remain live,
    /// and at the same address, for as long as the returned [`GraphExec`] can be
    /// replayed. Capture records raw device addresses, not Rust borrows, so
    /// nothing relates the two lifetimes and the compiler cannot help.
    ///
    /// This is `unsafe` because the violation is **silent**. Measured on an
    /// RTX A2000 (`tests/graph_capture.rs`, the `--ignored` probe), replaying a
    /// graph whose captured buffer had been dropped returned `Ok(())` from launch,
    /// synchronize *and* read-back, with the expected data — the replay read freed
    /// memory that still held the old bytes. Stream-ordered deallocation returning
    /// pages to a pool rather than the driver makes that the likely presentation,
    /// so the failure mode is "passes in testing, corrupts once pages are reused"
    /// rather than a fault. A safe signature would promise a check that neither
    /// the driver nor the type system performs.
    ///
    /// See the module docs for the three ways this could be closed; picking among
    /// them is a design decision, so this function states the obligation instead.
    pub unsafe fn instantiate(&self) -> Result<GraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut exec = MaybeUninit::<cuda_bindings::CUgraphExec>::uninit();
        let code = unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), self.cu_graph, 0)
        };
        Ok(GraphExec {
            cu_exec: (code, exec).result()?,
            ctx: Arc::clone(&self.ctx),
        })
    }

    /// Raw handle, for the parts of the graph surface this module does not wrap.
    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphDestroy(self.cu_graph);
        }
    }
}

/// An executable graph (`CUgraphExec`), ready for replay.
///
/// # Interaction with `launch_contract`
///
/// Kernels carrying a `requires` launch contract validate their size relations
/// **on the host, at launch**. Replay runs no host code, so those checks do not
/// execute per replay. That is sound as long as replay reuses the arguments
/// recorded at capture — each distinct configuration gets its own capture and
/// therefore its own validation — and the guarantee degrades from "every launch
/// is checked" to "this graph was validated when it was built". Mutating node
/// parameters after capture would break it, which is why this module does not
/// wrap `cuGraphExecKernelNodeSetParams`.
#[derive(Debug)]
pub struct GraphExec {
    cu_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
}

impl GraphExec {
    /// Replays the graph on `stream`.
    ///
    /// Takes `&mut self` because concurrent replays of one executable graph are
    /// not permitted; the borrow checker enforces the serialisation that the
    /// driver only documents.
    pub fn launch(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_exec, stream.cu_stream()).result() }
    }

    /// Raw handle, for the parts of the graph surface this module does not wrap.
    pub fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
        self.cu_exec
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(self.cu_exec);
        }
    }
}

impl CudaStream {
    /// Begins capturing work enqueued on this stream into a graph.
    ///
    /// The returned guard must be ended (or dropped) before the stream is used
    /// normally again.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<StreamCapture<'_>, DriverError> {
        self.context().bind_to_thread()?;
        unsafe {
            cuda_bindings::cuStreamBeginCapture_v2(self.cu_stream(), mode.as_raw()).result()?;
        }
        Ok(StreamCapture {
            stream: self,
            active: true,
        })
    }

    /// Whether this stream is currently capturing, and whether that capture has
    /// been invalidated by an error.
    pub fn capture_status(&self) -> Result<CaptureStatus, DriverError> {
        self.context().bind_to_thread()?;
        let mut status = MaybeUninit::<cuda_bindings::CUstreamCaptureStatus>::uninit();
        let code =
            unsafe { cuda_bindings::cuStreamIsCapturing(self.cu_stream(), status.as_mut_ptr()) };
        Ok(match (code, status).result()? {
            cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                CaptureStatus::Active
            }
            cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_INVALIDATED => {
                CaptureStatus::Invalidated
            }
            _ => CaptureStatus::None,
        })
    }
}

/// Capture state of a stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureStatus {
    /// Not capturing.
    None,
    /// Capturing normally.
    Active,
    /// Was capturing; an error invalidated the sequence. The capture must still
    /// be ended before the stream can be used again.
    Invalidated,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_mode_maps_to_driver_constants() {
        assert_eq!(
            CaptureMode::Global.as_raw(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL
        );
        assert_eq!(
            CaptureMode::ThreadLocal.as_raw(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        );
        assert_eq!(
            CaptureMode::Relaxed.as_raw(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED
        );
        // Global is the conservative choice, so it must be the default.
        assert_eq!(CaptureMode::default(), CaptureMode::Global);
    }
}

// ===========================================================================
// Experiment: can the buffer-lifetime obligation become a compile error?
// ===========================================================================

/// An executable graph whose lifetime is tied to the buffers its capture
/// referenced.
///
/// This is the experimental alternative to [`GraphExec`] + `unsafe instantiate`.
/// The idea: the driver never reports which allocations a captured graph
/// references, but it does not need to. What keeps a buffer alive is Rust's
/// *closure capture*, so if the returned exec carries a lifetime derived from
/// the closure, the borrow checker refuses to let the exec outlive anything the
/// closure borrowed.
///
/// Note there is **no runtime cost and no ownership change**: the closure runs
/// once during capture and is dropped immediately. Only the lifetime survives,
/// via `PhantomData`, and that is enough for the borrow checker to reject
///
/// ```ignore
/// let exec = {
///     let scratch = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
///     stream.capture_scoped(CaptureMode::Global, || { /* uses &scratch */ })?
/// };            // scratch dropped here
/// exec.launch(&stream)?;   // <- borrow checker error, not UB
/// ```
///
/// The rejection, pinned as a `compile_fail` doctest so the gate proves it rather
/// than the prose claiming it:
///
/// ```compile_fail
/// use cuda_core::graph::CaptureMode;
/// use cuda_core::{CudaContext, DeviceBuffer};
///
/// let ctx = CudaContext::new(0).unwrap();
/// let stream = ctx.new_stream().unwrap();
/// let mut dst = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
///
/// let mut exec = {
///     let src = DeviceBuffer::from_host(&stream, &[7u32, 8, 9, 10]).unwrap();
///     stream
///         .capture_scoped(CaptureMode::Global, || dst.copy_from_device_async(&src, &stream))
///         .unwrap()
///     // `src` dropped here -> E0597: `src` does not live long enough
/// };
/// exec.launch(&stream).unwrap();
/// ```
///
/// # The limitation that motivates [`OwnedGraphExec`]
///
/// The exec holds a *shared* borrow of everything the closure touched, so writing
/// new input between replays -- which is what graph replay is for -- does not
/// compile either:
///
/// ```compile_fail
/// use cuda_core::graph::CaptureMode;
/// use cuda_core::{CudaContext, DeviceBuffer};
///
/// let ctx = CudaContext::new(0).unwrap();
/// let stream = ctx.new_stream().unwrap();
/// let mut src = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
/// let mut dst = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
///
/// let mut exec = stream
///     .capture_scoped(CaptureMode::Global, || dst.copy_from_device_async(&src, &stream))
///     .unwrap();
///
/// // E0502: cannot borrow `src` as mutable because it is also borrowed as immutable
/// src.copy_from_host(&stream, &[1, 2, 3, 4]).unwrap();
/// exec.launch(&stream).unwrap();
/// ```
///
/// So this type is sound and zero-cost but rejects the primary use case. That is
/// the evidence for [`OwnedGraphExec`], not a guess.
#[derive(Debug)]
pub struct ScopedGraphExec<'captures> {
    cu_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
    /// Ties this exec to everything the capture closure borrowed. Invariant in
    /// `'captures` would be stricter than needed; covariance is correct here
    /// because a shorter-lived exec is always sound.
    captures: PhantomData<&'captures ()>,
}

impl ScopedGraphExec<'_> {
    /// Replays the graph. `&mut self` for the same reason as
    /// [`GraphExec::launch`]: concurrent replay of one exec is not permitted.
    pub fn launch(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_exec, stream.cu_stream()).result() }
    }
}

impl Drop for ScopedGraphExec<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(self.cu_exec);
        }
    }
}

impl CudaStream {
    /// Captures the work `record` enqueues, and returns an executable graph that
    /// **cannot outlive anything `record` borrowed**.
    ///
    /// This is safe where [`CapturedGraph::instantiate`] is `unsafe`, and the
    /// difference is the lifetime on the return type rather than any extra
    /// checking. The residual obligation is narrower but not empty: the work
    /// must reference device memory only through values the closure borrows. A
    /// raw pointer smuggled in from elsewhere is outside what the borrow checker
    /// can see, exactly as it would be anywhere else.
    ///
    /// The capture guard still terminates on the error path, so a `record` that
    /// returns `Err` leaves the stream usable.
    pub fn capture_scoped<'captures, F>(
        &self,
        mode: CaptureMode,
        record: F,
    ) -> Result<ScopedGraphExec<'captures>, DriverError>
    where
        F: FnOnce() -> Result<(), DriverError> + 'captures,
    {
        let capture = self.begin_capture(mode)?;
        // If `record` fails, dropping the guard terminates the capture, so the
        // stream is left usable rather than stuck in the invalidated state.
        record()?;
        let graph = capture.end()?;
        // SAFETY: the returned exec borrows `'captures`, so the borrow checker
        // will not let it outlive anything `record` referenced -- which is the
        // obligation `instantiate` otherwise pushes onto the caller.
        let exec = unsafe { graph.instantiate() }?;
        let cu_exec = exec.cu_graph_exec();
        // Hand the raw handle to the scoped wrapper and suppress the original
        // Drop, so ownership transfers rather than double-freeing.
        core::mem::forget(exec);
        Ok(ScopedGraphExec {
            cu_exec,
            ctx: Arc::clone(self.context()),
            captures: PhantomData,
        })
    }
}

/// An executable graph that **owns** the state its capture referenced, and lends
/// it back between replays.
///
/// This is the design the two simpler options fail to reach, and the failures are
/// worth recording because they are what motivates the shape:
///
/// * A lifetime-only exec ([`ScopedGraphExec`]) rejects the use-after-free at
///   compile time with no runtime cost — but it holds a shared borrow of every
///   captured buffer, so writing new input between replays is
///   `E0502: cannot borrow as mutable because it is also borrowed as immutable`.
///   Sound, and unusable for the thing graphs are for.
/// * `unsafe instantiate` with a documented obligation is usable and unsound: the
///   violation is silent (`Ok(())` from launch, sync and read, with plausible
///   data).
///
/// Moving the state in resolves both: nothing outside can drop it, and
/// [`state_mut`](Self::state_mut) hands it back for the write-then-replay loop.
///
/// # Stream ordering is the residual obligation
///
/// `launch` *enqueues* a replay; it does not wait for it. So a write issued after
/// `launch` returns is only safe if it is ordered behind the replay, which holds
/// when both go on the same stream — `DeviceBuffer`'s copy helpers are all
/// stream-ordered, so the natural usage is correct. Writing from a *different*
/// stream, or from the host without synchronising, races the in-flight graph.
/// That obligation is not expressible here and is documented rather than claimed.
#[derive(Debug)]
pub struct OwnedGraphExec<S> {
    cu_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
    state: S,
}

impl<S> OwnedGraphExec<S> {
    /// Replays the graph on `stream`.
    pub fn launch(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_exec, stream.cu_stream()).result() }
    }

    /// The captured state, for writing new input before the next replay.
    ///
    /// `&mut self` means this cannot overlap a `launch` call, though see the note
    /// on stream ordering: it does not order against a replay already in flight.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// The captured state, immutably.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Gives the captured state back, destroying the graph.
    pub fn into_state(self) -> S {
        // Move `state` out without running `Drop for OwnedGraphExec`, then free
        // the exec by hand so the graph is not leaked.
        let me = core::mem::ManuallyDrop::new(self);
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(me.cu_exec);
            core::ptr::read(&me.state)
        }
    }
}

impl<S> Drop for OwnedGraphExec<S> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(self.cu_exec);
        }
    }
}

impl CudaStream {
    /// Captures work that operates on `state`, and returns an executable graph
    /// owning it.
    ///
    /// `record` receives `&mut S` and must enqueue its work on this stream. The
    /// state is then reachable through [`OwnedGraphExec::state_mut`] for the
    /// write-then-replay loop, and recoverable with
    /// [`OwnedGraphExec::into_state`].
    ///
    /// Safe, because nothing outside the exec can drop the buffers the capture
    /// referenced. See the type docs for the stream-ordering obligation that
    /// remains.
    pub fn capture_owning<S, F>(
        &self,
        mode: CaptureMode,
        mut state: S,
        record: F,
    ) -> Result<OwnedGraphExec<S>, DriverError>
    where
        F: FnOnce(&mut S) -> Result<(), DriverError>,
    {
        let capture = self.begin_capture(mode)?;
        // On `Err`, dropping the guard terminates the capture, so the stream
        // stays usable and `state` is returned to the caller by being dropped.
        record(&mut state)?;
        let graph = capture.end()?;
        // SAFETY: `state` is moved into the returned exec, so every buffer the
        // capture referenced through it outlives every replay by construction.
        let exec = unsafe { graph.instantiate() }?;
        let cu_exec = exec.cu_graph_exec();
        core::mem::forget(exec);
        Ok(OwnedGraphExec {
            cu_exec,
            ctx: Arc::clone(self.context()),
            state,
        })
    }
}
