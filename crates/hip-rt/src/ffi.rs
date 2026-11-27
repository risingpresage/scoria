#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::mem::{size_of, transmute_copy};
use std::sync::OnceLock;

pub type hipError_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipGraph_t = *mut c_void;
pub type hipGraphExec_t = *mut c_void;
pub type hipGraphNode_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;

pub const HIP_MEMCPY_HOST_TO_HOST: c_int = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0x0;
pub const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;
/// `hipStreamCaptureModeGlobal`.
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 87;
pub const HIP_DEVICE_PROP_R0600_SIZE: usize = 1472;
pub const HIP_DEVICE_PROP_R0600_GCN_ARCH_OFFSET: usize = 1160;

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

pub struct HipApi {
    _handle: *mut c_void,
    pub hipSetDevice: unsafe extern "C" fn(deviceId: c_int) -> hipError_t,
    pub hipDeviceSynchronize: unsafe extern "C" fn() -> hipError_t,
    pub hipGetErrorString: unsafe extern "C" fn(hipError: hipError_t) -> *const c_char,
    pub hipMemGetInfo: unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> hipError_t,
    pub hipDeviceGetAttribute:
        unsafe extern "C" fn(pi: *mut c_int, attr: c_int, deviceId: c_int) -> hipError_t,
    pub hipGetDevicePropertiesR0600:
        unsafe extern "C" fn(prop: *mut u8, deviceId: c_int) -> hipError_t,

    pub hipMalloc: unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> hipError_t,
    pub hipFree: unsafe extern "C" fn(ptr: *mut c_void) -> hipError_t,
    pub hipHostMalloc:
        unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t,
    pub hipHostFree: unsafe extern "C" fn(ptr: *mut c_void) -> hipError_t,

    pub hipMemcpy: unsafe extern "C" fn(
        dst: *mut c_void,
        src: *const c_void,
        sizeBytes: usize,
        kind: c_int,
    ) -> hipError_t,
    pub hipMemcpyAsync: unsafe extern "C" fn(
        dst: *mut c_void,
        src: *const c_void,
        sizeBytes: usize,
        kind: c_int,
        stream: hipStream_t,
    ) -> hipError_t,
    pub hipMemsetAsync: unsafe extern "C" fn(
        dst: *mut c_void,
        value: c_int,
        sizeBytes: usize,
        stream: hipStream_t,
    ) -> hipError_t,

    pub hipStreamCreateWithFlags:
        unsafe extern "C" fn(stream: *mut hipStream_t, flags: c_uint) -> hipError_t,
    pub hipStreamSynchronize: unsafe extern "C" fn(stream: hipStream_t) -> hipError_t,
    pub hipStreamDestroy: unsafe extern "C" fn(stream: hipStream_t) -> hipError_t,

    pub hipEventCreate: unsafe extern "C" fn(event: *mut hipEvent_t) -> hipError_t,
    pub hipEventRecord: unsafe extern "C" fn(event: hipEvent_t, stream: hipStream_t) -> hipError_t,
    pub hipEventSynchronize: unsafe extern "C" fn(event: hipEvent_t) -> hipError_t,
    pub hipEventElapsedTime:
        unsafe extern "C" fn(ms: *mut f32, start: hipEvent_t, stop: hipEvent_t) -> hipError_t,
    pub hipEventDestroy: unsafe extern "C" fn(event: hipEvent_t) -> hipError_t,

    pub hipModuleLoadData:
        unsafe extern "C" fn(module: *mut hipModule_t, image: *const c_void) -> hipError_t,
    pub hipModuleGetFunction: unsafe extern "C" fn(
        function: *mut hipFunction_t,
        module: hipModule_t,
        kname: *const c_char,
    ) -> hipError_t,
    pub hipModuleLaunchKernel: unsafe extern "C" fn(
        f: hipFunction_t,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        stream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t,
    pub hipModuleUnload: unsafe extern "C" fn(module: hipModule_t) -> hipError_t,

    pub hipStreamBeginCapture: unsafe extern "C" fn(stream: hipStream_t, mode: c_int) -> hipError_t,
    pub hipStreamEndCapture:
        unsafe extern "C" fn(stream: hipStream_t, graph: *mut hipGraph_t) -> hipError_t,
    pub hipGraphInstantiate: unsafe extern "C" fn(
        exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        errorNode: *mut hipGraphNode_t,
        logBuffer: *mut c_char,
        bufferSize: usize,
    ) -> hipError_t,
    pub hipGraphLaunch:
        unsafe extern "C" fn(exec: hipGraphExec_t, stream: hipStream_t) -> hipError_t,
    pub hipGraphDestroy: unsafe extern "C" fn(graph: hipGraph_t) -> hipError_t,
    pub hipGraphExecDestroy: unsafe extern "C" fn(exec: hipGraphExec_t) -> hipError_t,
}

unsafe impl Send for HipApi {}
unsafe impl Sync for HipApi {}

unsafe fn load_sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
    assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
    let cname = CString::new(name).map_err(|e| e.to_string())?;
    let ptr = dlsym(handle, cname.as_ptr());
    if ptr.is_null() {
        let err = dlerror();
        let msg = if err.is_null() {
            "symbol not found".to_string()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        };
        return Err(format!("dlsym({name}) failed: {msg}"));
    }
    Ok(transmute_copy::<*mut c_void, T>(&ptr))
}

unsafe fn load_hip_api() -> Result<HipApi, String> {
    let mut candidates = Vec::new();
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        candidates.push(format!("{rocm_path}/lib/libamdhip64.so"));
    }
    candidates.push("/opt/rocm/lib/libamdhip64.so".to_string());
    candidates.push("libamdhip64.so.7".to_string());
    candidates.push("libamdhip64.so".to_string());

    let mut handle = std::ptr::null_mut();
    let mut last_err = String::new();
    for cand in &candidates {
        let cname = CString::new(cand.as_str()).map_err(|e| e.to_string())?;
        handle = dlopen(cname.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        if !handle.is_null() {
            break;
        }
        let err = dlerror();
        if !err.is_null() {
            last_err = CStr::from_ptr(err).to_string_lossy().into_owned();
        }
    }
    if handle.is_null() {
        return Err(format!(
            "failed to dlopen libamdhip64.so (last error: {last_err})"
        ));
    }

    Ok(HipApi {
        _handle: handle,
        hipSetDevice: load_sym(handle, "hipSetDevice")?,
        hipDeviceSynchronize: load_sym(handle, "hipDeviceSynchronize")?,
        hipGetErrorString: load_sym(handle, "hipGetErrorString")?,
        hipMemGetInfo: load_sym(handle, "hipMemGetInfo")?,
        hipDeviceGetAttribute: load_sym(handle, "hipDeviceGetAttribute")?,
        hipGetDevicePropertiesR0600: load_sym(handle, "hipGetDevicePropertiesR0600")?,
        hipMalloc: load_sym(handle, "hipMalloc")?,
        hipFree: load_sym(handle, "hipFree")?,
        hipHostMalloc: load_sym(handle, "hipHostMalloc")?,
        hipHostFree: load_sym(handle, "hipHostFree")?,
        hipMemcpy: load_sym(handle, "hipMemcpy")?,
        hipMemcpyAsync: load_sym(handle, "hipMemcpyAsync")?,
        hipMemsetAsync: load_sym(handle, "hipMemsetAsync")?,
        hipStreamCreateWithFlags: load_sym(handle, "hipStreamCreateWithFlags")?,
        hipStreamSynchronize: load_sym(handle, "hipStreamSynchronize")?,
        hipStreamDestroy: load_sym(handle, "hipStreamDestroy")?,
        hipEventCreate: load_sym(handle, "hipEventCreate")?,
        hipEventRecord: load_sym(handle, "hipEventRecord")?,
        hipEventSynchronize: load_sym(handle, "hipEventSynchronize")?,
        hipEventElapsedTime: load_sym(handle, "hipEventElapsedTime")?,
        hipEventDestroy: load_sym(handle, "hipEventDestroy")?,
        hipModuleLoadData: load_sym(handle, "hipModuleLoadData")?,
        hipModuleGetFunction: load_sym(handle, "hipModuleGetFunction")?,
        hipModuleLaunchKernel: load_sym(handle, "hipModuleLaunchKernel")?,
        hipModuleUnload: load_sym(handle, "hipModuleUnload")?,
        hipStreamBeginCapture: load_sym(handle, "hipStreamBeginCapture")?,
        hipStreamEndCapture: load_sym(handle, "hipStreamEndCapture")?,
        hipGraphInstantiate: load_sym(handle, "hipGraphInstantiate")?,
        hipGraphLaunch: load_sym(handle, "hipGraphLaunch")?,
        hipGraphDestroy: load_sym(handle, "hipGraphDestroy")?,
        hipGraphExecDestroy: load_sym(handle, "hipGraphExecDestroy")?,
    })
}

static HIP_API: OnceLock<Result<HipApi, String>> = OnceLock::new();

pub fn api() -> Result<&'static HipApi, &'static str> {
    HIP_API
        .get_or_init(|| unsafe { load_hip_api() })
        .as_ref()
        .map_err(|e| e.as_str())
}
