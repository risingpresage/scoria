mod alloc_guard;
mod ffi;

use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::ptr;

use ffi::*;

#[derive(Debug, Clone)]
pub struct HipError {
    pub code: i32,
    pub message: String,
    pub op: &'static str,
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HIP {} failed ({}): {}",
            self.op, self.code, self.message
        )
    }
}

impl std::error::Error for HipError {}

pub type HipResult<T> = Result<T, HipError>;

fn get_api(op: &'static str) -> HipResult<&'static HipApi> {
    alloc_guard::ensure_linked();
    api().map_err(|msg| HipError {
        code: -1,
        message: msg.to_string(),
        op,
    })
}

fn check(hip: &HipApi, code: hipError_t, op: &'static str) -> HipResult<()> {
    if code == HIP_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let p = (hip.hipGetErrorString)(code);
        if p.is_null() {
            "unknown HIP error".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(HipError {
        code,
        message: msg,
        op,
    })
}

pub fn set_device(device_id: i32) -> HipResult<()> {
    let hip = get_api("hipSetDevice")?;
    check(
        hip,
        unsafe { (hip.hipSetDevice)(device_id) },
        "hipSetDevice",
    )
}

pub fn device_synchronize() -> HipResult<()> {
    let hip = get_api("hipDeviceSynchronize")?;
    check(
        hip,
        unsafe { (hip.hipDeviceSynchronize)() },
        "hipDeviceSynchronize",
    )
}

pub fn mem_info() -> HipResult<(usize, usize)> {
    let hip = get_api("hipMemGetInfo")?;
    let mut free = 0usize;
    let mut total = 0usize;
    check(
        hip,
        unsafe { (hip.hipMemGetInfo)(&mut free, &mut total) },
        "hipMemGetInfo",
    )?;
    Ok((free, total))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetArch {
    Gfx1201,
    Gfx942,
    Gfx950,
    Other(String),
}

impl TargetArch {
    pub fn as_str(&self) -> &str {
        match self {
            TargetArch::Gfx1201 => "gfx1201",
            TargetArch::Gfx942 => "gfx942",
            TargetArch::Gfx950 => "gfx950",
            TargetArch::Other(s) => s.as_str(),
        }
    }

    pub fn is_rdna4(&self) -> bool {
        matches!(self, TargetArch::Gfx1201)
    }
}

pub fn warp_size(device_id: i32) -> HipResult<u32> {
    let hip = get_api("hipDeviceGetAttribute(WarpSize)")?;
    let mut ws = 0i32;
    check(
        hip,
        unsafe { (hip.hipDeviceGetAttribute)(&mut ws, HIP_DEVICE_ATTRIBUTE_WARP_SIZE, device_id) },
        "hipDeviceGetAttribute(WarpSize)",
    )?;
    Ok(ws as u32)
}

pub fn detect_arch(device_id: i32) -> HipResult<TargetArch> {
    let hip = get_api("hipGetDevicePropertiesR0600")?;
    let mut prop = vec![0u8; HIP_DEVICE_PROP_R0600_SIZE];
    check(
        hip,
        unsafe { (hip.hipGetDevicePropertiesR0600)(prop.as_mut_ptr(), device_id) },
        "hipGetDevicePropertiesR0600",
    )?;
    let slice =
        &prop[HIP_DEVICE_PROP_R0600_GCN_ARCH_OFFSET..HIP_DEVICE_PROP_R0600_GCN_ARCH_OFFSET + 256];
    let nul = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    let full = String::from_utf8_lossy(&slice[..nul]);
    let base = full.split(':').next().unwrap_or(full.as_ref());
    Ok(match base {
        "gfx1201" | "gfx1200" => TargetArch::Gfx1201,
        "gfx942" => TargetArch::Gfx942,
        "gfx950" => TargetArch::Gfx950,
        other => TargetArch::Other(other.to_string()),
    })
}

pub struct DeviceBuf {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for DeviceBuf {}
unsafe impl Sync for DeviceBuf {}

impl DeviceBuf {
    pub fn alloc(bytes: usize) -> HipResult<Self> {
        let hip = get_api("hipMalloc")?;
        let mut ptr = ptr::null_mut();
        let alloc_bytes = bytes.max(16);
        check(
            hip,
            unsafe { (hip.hipMalloc)(&mut ptr, alloc_bytes) },
            "hipMalloc",
        )?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn as_ptr_mut(&mut self) -> *mut c_void {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn copy_from_host(&mut self, src: &[u8]) -> HipResult<()> {
        assert!(src.len() <= self.bytes, "H2D copy exceeds device buffer");
        let hip = get_api("hipMemcpy H2D")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpy)(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy H2D",
        )
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) -> HipResult<()> {
        assert!(dst.len() <= self.bytes, "D2H copy exceeds device buffer");
        let hip = get_api("hipMemcpy D2H")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    dst.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy D2H",
        )
    }

    pub fn copy_from_host_async(
        &mut self,
        src: *const u8,
        bytes: usize,
        stream: &HipStream,
    ) -> HipResult<()> {
        assert!(bytes <= self.bytes, "async H2D copy exceeds device buffer");
        let hip = get_api("hipMemcpyAsync H2D")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpyAsync)(
                    self.ptr,
                    src as *const c_void,
                    bytes,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync H2D",
        )
    }

    pub fn copy_to_host_async(
        &self,
        dst: *mut u8,
        bytes: usize,
        stream: &HipStream,
    ) -> HipResult<()> {
        assert!(bytes <= self.bytes, "async D2H copy exceeds device buffer");
        let hip = get_api("hipMemcpyAsync D2H")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpyAsync)(
                    dst as *mut c_void,
                    self.ptr,
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync D2H",
        )
    }

    pub fn zero_async(&mut self, stream: &HipStream) -> HipResult<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        let hip = get_api("hipMemsetAsync")?;
        check(
            hip,
            unsafe { (hip.hipMemsetAsync)(self.ptr, 0, self.bytes, stream.raw()) },
            "hipMemsetAsync",
        )
    }

    pub fn copy_from_host_at(&mut self, dst_byte_offset: usize, src: &[u8]) -> HipResult<()> {
        assert!(
            dst_byte_offset + src.len() <= self.bytes,
            "offset H2D copy exceeds device buffer"
        );
        if src.is_empty() {
            return Ok(());
        }
        let hip = get_api("hipMemcpy H2D offset")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpy)(
                    (self.ptr as *mut u8).add(dst_byte_offset) as *mut c_void,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy H2D offset",
        )
    }

    pub fn copy_to_host_at(&self, src_byte_offset: usize, dst: &mut [u8]) -> HipResult<()> {
        assert!(
            src_byte_offset + dst.len() <= self.bytes,
            "offset D2H copy exceeds device buffer"
        );
        if dst.is_empty() {
            return Ok(());
        }
        let hip = get_api("hipMemcpy D2H offset")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    (self.ptr as *const u8).add(src_byte_offset) as *const c_void,
                    dst.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy D2H offset",
        )
    }

    pub fn copy_from_device_async(
        &mut self,
        dst_byte_offset: usize,
        src: &DeviceBuf,
        src_byte_offset: usize,
        bytes: usize,
        stream: &HipStream,
    ) -> HipResult<()> {
        assert!(
            dst_byte_offset + bytes <= self.bytes && src_byte_offset + bytes <= src.bytes,
            "async D2D copy exceeds device buffer"
        );
        if bytes == 0 {
            return Ok(());
        }
        let hip = get_api("hipMemcpyAsync D2D")?;
        check(
            hip,
            unsafe {
                (hip.hipMemcpyAsync)(
                    (self.ptr as *mut u8).add(dst_byte_offset) as *mut c_void,
                    (src.ptr as *const u8).add(src_byte_offset) as *const c_void,
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync D2D",
        )
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(hip) = get_api("hipFree") {
                unsafe {
                    let _ = (hip.hipFree)(self.ptr);
                }
            }
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct PinnedBuf {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    pub fn alloc(bytes: usize) -> HipResult<Self> {
        let hip = get_api("hipHostMalloc")?;
        let mut ptr = ptr::null_mut();
        let alloc_bytes = bytes.max(16);
        check(
            hip,
            unsafe { (hip.hipHostMalloc)(&mut ptr, alloc_bytes, HIP_HOST_MALLOC_DEFAULT) },
            "hipHostMalloc",
        )?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.bytes) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(hip) = get_api("hipHostFree") {
                unsafe {
                    let _ = (hip.hipHostFree)(self.ptr);
                }
            }
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct HipStream {
    raw: hipStream_t,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn new() -> HipResult<Self> {
        let hip = get_api("hipStreamCreateWithFlags")?;
        let mut raw = ptr::null_mut();
        check(
            hip,
            unsafe { (hip.hipStreamCreateWithFlags)(&mut raw, HIP_STREAM_NON_BLOCKING) },
            "hipStreamCreateWithFlags",
        )?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> hipStream_t {
        self.raw
    }

    pub fn synchronize(&self) -> HipResult<()> {
        let hip = get_api("hipStreamSynchronize")?;
        check(
            hip,
            unsafe { (hip.hipStreamSynchronize)(self.raw) },
            "hipStreamSynchronize",
        )
    }

    /// Begins stream capture. Everything submitted to this stream until
    /// [`HipStream::end_capture`] becomes a replayable [`HipGraph`].
    pub fn begin_capture(&self) -> HipResult<()> {
        let hip = get_api("hipStreamBeginCapture")?;
        check(
            hip,
            unsafe { (hip.hipStreamBeginCapture)(self.raw, HIP_STREAM_CAPTURE_MODE_GLOBAL) },
            "hipStreamBeginCapture",
        )
    }

    pub fn end_capture(&self) -> HipResult<HipGraph> {
        let hip = get_api("hipStreamEndCapture")?;
        let mut raw = ptr::null_mut();
        check(
            hip,
            unsafe { (hip.hipStreamEndCapture)(self.raw, &mut raw) },
            "hipStreamEndCapture",
        )?;
        Ok(HipGraph { raw })
    }
}

/// A captured, not-yet-instantiated HIP graph.
pub struct HipGraph {
    raw: hipGraph_t,
}

unsafe impl Send for HipGraph {}
unsafe impl Sync for HipGraph {}

impl HipGraph {
    pub fn instantiate(&self) -> HipResult<HipGraphExec> {
        let hip = get_api("hipGraphInstantiate")?;
        let mut exec = ptr::null_mut();
        let mut log = vec![0u8; 2048];
        let mut err_node: hipGraphNode_t = ptr::null_mut();
        check(
            hip,
            unsafe {
                (hip.hipGraphInstantiate)(
                    &mut exec,
                    self.raw,
                    &mut err_node,
                    log.as_mut_ptr() as *mut std::ffi::c_char,
                    log.len(),
                )
            },
            "hipGraphInstantiate",
        )?;
        Ok(HipGraphExec { raw: exec })
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(hip) = get_api("hipGraphDestroy") {
                unsafe {
                    let _ = (hip.hipGraphDestroy)(self.raw);
                }
            }
            self.raw = ptr::null_mut();
        }
    }
}

/// An instantiated graph, ready to be replayed on any stream.
pub struct HipGraphExec {
    raw: hipGraphExec_t,
}

unsafe impl Send for HipGraphExec {}
unsafe impl Sync for HipGraphExec {}

impl HipGraphExec {
    pub fn launch(&self, stream: &HipStream) -> HipResult<()> {
        let hip = get_api("hipGraphLaunch")?;
        check(
            hip,
            unsafe { (hip.hipGraphLaunch)(self.raw, stream.raw()) },
            "hipGraphLaunch",
        )
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(hip) = get_api("hipGraphExecDestroy") {
                unsafe {
                    let _ = (hip.hipGraphExecDestroy)(self.raw);
                }
            }
            self.raw = ptr::null_mut();
        }
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(hip) = get_api("hipStreamDestroy") {
                unsafe {
                    let _ = (hip.hipStreamDestroy)(self.raw);
                }
            }
            self.raw = ptr::null_mut();
        }
    }
}

pub struct HipEvent {
    raw: hipEvent_t,
}

unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    pub fn new() -> HipResult<Self> {
        let hip = get_api("hipEventCreate")?;
        let mut raw = ptr::null_mut();
        check(
            hip,
            unsafe { (hip.hipEventCreate)(&mut raw) },
            "hipEventCreate",
        )?;
        Ok(Self { raw })
    }

    pub fn record(&self, stream: &HipStream) -> HipResult<()> {
        let hip = get_api("hipEventRecord")?;
        check(
            hip,
            unsafe { (hip.hipEventRecord)(self.raw, stream.raw()) },
            "hipEventRecord",
        )
    }

    pub fn synchronize(&self) -> HipResult<()> {
        let hip = get_api("hipEventSynchronize")?;
        check(
            hip,
            unsafe { (hip.hipEventSynchronize)(self.raw) },
            "hipEventSynchronize",
        )
    }

    pub fn elapsed_ms(start: &HipEvent, stop: &HipEvent) -> HipResult<f32> {
        let hip = get_api("hipEventElapsedTime")?;
        let mut ms = 0.0f32;
        check(
            hip,
            unsafe { (hip.hipEventElapsedTime)(&mut ms, start.raw, stop.raw) },
            "hipEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(hip) = get_api("hipEventDestroy") {
                unsafe {
                    let _ = (hip.hipEventDestroy)(self.raw);
                }
            }
            self.raw = ptr::null_mut();
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KernelHandle {
    raw: hipFunction_t,
}

unsafe impl Send for KernelHandle {}
unsafe impl Sync for KernelHandle {}

pub struct HipModule {
    raw: hipModule_t,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    pub fn from_hsaco(hsaco: &[u8]) -> HipResult<Self> {
        let hip = get_api("hipModuleLoadData")?;
        let mut raw = ptr::null_mut();
        check(
            hip,
            unsafe { (hip.hipModuleLoadData)(&mut raw, hsaco.as_ptr() as *const c_void) },
            "hipModuleLoadData",
        )?;
        Ok(Self { raw })
    }

    pub fn resolve(&self, name: &str) -> HipResult<KernelHandle> {
        let hip = get_api("hipModuleGetFunction")?;
        let cname = CString::new(name).expect("kernel name must not contain NUL");
        let mut func = ptr::null_mut();
        check(
            hip,
            unsafe { (hip.hipModuleGetFunction)(&mut func, self.raw, cname.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(KernelHandle { raw: func })
    }

    /// Launches a pre-resolved kernel handle tied to this module.
    ///
    /// # Safety
    /// Callers must ensure `handle` was resolved from `self`, `params` pointers match the device
    /// kernel signature, and all buffers outlive execution.
    pub unsafe fn launch_handle(
        &self,
        handle: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        stream: &HipStream,
        params: &mut [*mut c_void],
    ) -> HipResult<()> {
        let hip = get_api("hipModuleLaunchKernel")?;
        check(
            hip,
            (hip.hipModuleLaunchKernel)(
                handle.raw,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem_bytes,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "hipModuleLaunchKernel",
        )
    }

    pub fn function(&self, name: &str) -> HipResult<HipFunction<'_>> {
        let hip = get_api("hipModuleGetFunction")?;
        let cname = CString::new(name).expect("kernel name must not contain NUL");
        let mut func = ptr::null_mut();
        check(
            hip,
            unsafe { (hip.hipModuleGetFunction)(&mut func, self.raw, cname.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(HipFunction {
            raw: func,
            _module: self,
        })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(hip) = get_api("hipModuleUnload") {
                unsafe {
                    let _ = (hip.hipModuleUnload)(self.raw);
                }
            }
            self.raw = ptr::null_mut();
        }
    }
}

pub struct HipFunction<'a> {
    raw: hipFunction_t,
    _module: &'a HipModule,
}

impl<'a> HipFunction<'a> {
    /// Launches the kernel.
    ///
    /// # Safety
    /// Callers must ensure `params` pointers match the device kernel signature and outlive execution.
    pub unsafe fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        stream: &HipStream,
        params: &mut [*mut c_void],
    ) -> HipResult<()> {
        let hip = get_api("hipModuleLaunchKernel")?;
        check(
            hip,
            (hip.hipModuleLaunchKernel)(
                self.raw,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem_bytes,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "hipModuleLaunchKernel",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_gfx1201_arch_and_wave32() {
        set_device(0).expect("hipSetDevice(0)");
        assert_eq!(detect_arch(0).expect("detect_arch"), TargetArch::Gfx1201);
        assert_eq!(warp_size(0).expect("warp_size"), 32);
    }

    #[test]
    fn roundtrip_device_and_pinned_4mib() {
        set_device(0).expect("hipSetDevice(0)");
        let len = 4 * 1024 * 1024;
        let mut pinned = PinnedBuf::alloc(len).expect("hipHostMalloc");
        for (i, b) in pinned.as_mut_slice().iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xff) as u8;
        }
        let mut dev = DeviceBuf::alloc(len).expect("hipMalloc");
        dev.copy_from_host(pinned.as_slice()).expect("H2D");
        let mut out = vec![0u8; len];
        dev.copy_to_host(&mut out).expect("D2H");
        assert_eq!(pinned.as_slice(), out.as_slice());
    }
}
