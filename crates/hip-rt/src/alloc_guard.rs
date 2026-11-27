#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod imp {
    use std::ffi::c_void;
    use std::ptr;

    extern "C" {
        fn __libc_memalign(align: usize, size: usize) -> *mut c_void;
        fn __libc_free(ptr: *mut c_void);
        fn malloc_usable_size(ptr: *mut c_void) -> usize;
    }

    /// 32-byte aligned `malloc` override for glibc to satisfy ROCm `vmovdqa %ymm` stores.
    ///
    /// # Safety
    /// Follows C `malloc` ABI contract.
    #[no_mangle]
    pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
        __libc_memalign(32, size.max(1))
    }

    /// 32-byte aligned `calloc` override for glibc.
    ///
    /// # Safety
    /// Follows C `calloc` ABI contract.
    #[no_mangle]
    pub unsafe extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
        let Some(total) = nmemb.checked_mul(size) else {
            return ptr::null_mut();
        };
        let p = __libc_memalign(32, total.max(1));
        if !p.is_null() && total > 0 {
            ptr::write_bytes(p as *mut u8, 0, total);
        }
        p
    }

    /// 32-byte aligned `realloc` override for glibc.
    ///
    /// # Safety
    /// `ptr` must be null or previously returned by this allocator.
    #[no_mangle]
    pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
        if ptr.is_null() {
            return malloc(size);
        }
        if size == 0 {
            __libc_free(ptr);
            return ptr::null_mut();
        }
        let old_sz = malloc_usable_size(ptr);
        if (ptr as usize).is_multiple_of(32) && size <= old_sz {
            return ptr;
        }
        let np = __libc_memalign(32, size);
        if np.is_null() {
            return ptr::null_mut();
        }
        ptr::copy_nonoverlapping(ptr as *const u8, np as *mut u8, old_sz.min(size));
        __libc_free(ptr);
        np
    }

    pub fn ensure_linked() {
        std::hint::black_box((
            malloc as *const (),
            calloc as *const (),
            realloc as *const (),
        ));
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub use imp::ensure_linked;

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn ensure_linked() {}
