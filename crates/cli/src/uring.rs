#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_int, c_long, c_void, CString};
use std::mem::{size_of, zeroed};
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const SYS_IO_URING_SETUP: c_long = 425;
const SYS_IO_URING_ENTER: c_long = 426;

const IORING_OP_READ: u8 = 22;
const IORING_ENTER_GETEVENTS: u32 = 1;
const IORING_FEAT_SINGLE_MMAP: u32 = 1;

const IORING_OFF_SQ_RING: i64 = 0;
const IORING_OFF_CQ_RING: i64 = 0x800_0000;
const IORING_OFF_SQES: i64 = 0x1000_0000;

const O_RDONLY: c_int = 0;
const O_DIRECT: c_int = 0x4000;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
const MAP_POPULATE: c_int = 0x8000;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IoSqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IoCqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IoUringSqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    off: u64,
    addr: u64,
    len: u32,
    rw_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    addr3: u64,
    pad2: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct IoUringCqe {
    pub user_data: u64,
    pub res: i32,
    pub flags: u32,
}

const _: () = assert!(size_of::<IoUringParams>() == 120);
const _: () = assert!(size_of::<IoUringSqe>() == 64);
const _: () = assert!(size_of::<IoUringCqe>() == 16);

extern "C" {
    fn syscall(num: c_long, ...) -> c_long;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    fn open(pathname: *const i8, flags: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int;
    fn free(ptr: *mut c_void);
    fn __errno_location() -> *mut c_int;
}

fn last_errno() -> i32 {
    unsafe { *__errno_location() }
}

pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    pub fn alloc(align: usize, len: usize) -> Result<Self, String> {
        let mut raw: *mut c_void = ptr::null_mut();
        let rc = unsafe { posix_memalign(&mut raw, align, len) };
        if rc != 0 || raw.is_null() {
            return Err(format!("posix_memalign({align}, {len}) failed: {rc}"));
        }
        unsafe { ptr::write_bytes(raw as *mut u8, 0, len) };
        Ok(Self {
            ptr: raw as *mut u8,
            len,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { free(self.ptr as *mut c_void) };
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct IoUring {
    ring_fd: c_int,
    sq_ptr: *mut u8,
    sq_len: usize,
    cq_ptr: *mut u8,
    cq_len: usize,
    single_mmap: bool,
    sqes: *mut IoUringSqe,
    sqes_len: usize,
    sq_head: *const AtomicU32,
    sq_tail: *mut AtomicU32,
    sq_mask: u32,
    sq_array: *mut u32,
    cq_head: *mut AtomicU32,
    cq_tail: *const AtomicU32,
    cq_mask: u32,
    cqes: *const IoUringCqe,
}

impl IoUring {
    pub fn new(entries: u32) -> Result<Self, String> {
        let mut params: IoUringParams = unsafe { zeroed() };
        let ring_fd = unsafe {
            syscall(
                SYS_IO_URING_SETUP,
                entries as c_long,
                &mut params as *mut IoUringParams,
            )
        };
        if ring_fd < 0 {
            return Err(format!("io_uring_setup failed: errno {}", last_errno()));
        }
        let ring_fd = ring_fd as c_int;
        let single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;

        let mut sq_len =
            params.sq_off.array as usize + params.sq_entries as usize * size_of::<u32>();
        let cq_len =
            params.cq_off.cqes as usize + params.cq_entries as usize * size_of::<IoUringCqe>();
        if single_mmap && cq_len > sq_len {
            sq_len = cq_len;
        }

        let sq_ptr = unsafe {
            mmap(
                ptr::null_mut(),
                sq_len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_POPULATE,
                ring_fd,
                IORING_OFF_SQ_RING,
            )
        };
        if sq_ptr as isize == -1 {
            let err = last_errno();
            unsafe { close(ring_fd) };
            return Err(format!("mmap sq_ring failed: errno {err}"));
        }
        let sq_ptr = sq_ptr as *mut u8;

        let cq_ptr = if single_mmap {
            sq_ptr
        } else {
            let p = unsafe {
                mmap(
                    ptr::null_mut(),
                    cq_len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE,
                    ring_fd,
                    IORING_OFF_CQ_RING,
                )
            };
            if p as isize == -1 {
                let err = last_errno();
                unsafe {
                    munmap(sq_ptr as *mut c_void, sq_len);
                    close(ring_fd);
                }
                return Err(format!("mmap cq_ring failed: errno {err}"));
            }
            p as *mut u8
        };

        let sqes_len = params.sq_entries as usize * size_of::<IoUringSqe>();
        let sqes_raw = unsafe {
            mmap(
                ptr::null_mut(),
                sqes_len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_POPULATE,
                ring_fd,
                IORING_OFF_SQES,
            )
        };
        if sqes_raw as isize == -1 {
            let err = last_errno();
            unsafe {
                if !single_mmap {
                    munmap(cq_ptr as *mut c_void, cq_len);
                }
                munmap(sq_ptr as *mut c_void, sq_len);
                close(ring_fd);
            }
            return Err(format!("mmap sqes failed: errno {err}"));
        }

        unsafe {
            let sq_head = sq_ptr.add(params.sq_off.head as usize) as *const AtomicU32;
            let sq_tail = sq_ptr.add(params.sq_off.tail as usize) as *mut AtomicU32;
            let sq_mask = *(sq_ptr.add(params.sq_off.ring_mask as usize) as *const u32);
            let sq_array = sq_ptr.add(params.sq_off.array as usize) as *mut u32;

            let cq_head = cq_ptr.add(params.cq_off.head as usize) as *mut AtomicU32;
            let cq_tail = cq_ptr.add(params.cq_off.tail as usize) as *const AtomicU32;
            let cq_mask = *(cq_ptr.add(params.cq_off.ring_mask as usize) as *const u32);
            let cqes = cq_ptr.add(params.cq_off.cqes as usize) as *const IoUringCqe;

            Ok(Self {
                ring_fd,
                sq_ptr,
                sq_len,
                cq_ptr,
                cq_len,
                single_mmap,
                sqes: sqes_raw as *mut IoUringSqe,
                sqes_len,
                sq_head,
                sq_tail,
                sq_mask,
                sq_array,
                cq_head,
                cq_tail,
                cq_mask,
                cqes,
            })
        }
    }

    pub fn submit_and_wait(&self, to_submit: u32, min_complete: u32) -> Result<u32, String> {
        let rc = unsafe {
            syscall(
                SYS_IO_URING_ENTER,
                self.ring_fd as c_long,
                to_submit as c_long,
                min_complete as c_long,
                IORING_ENTER_GETEVENTS as c_long,
                ptr::null::<c_void>(),
                0usize as c_long,
            )
        };
        if rc < 0 {
            return Err(format!("io_uring_enter failed: errno {}", last_errno()));
        }
        Ok(rc as u32)
    }

    pub fn push_read(&self, fd: i32, offset: u64, dst: *mut u8, len: u32, user_data: u64) -> bool {
        unsafe {
            let tail = (*self.sq_tail).load(Ordering::Relaxed);
            let head = (*self.sq_head).load(Ordering::Acquire);
            if tail.wrapping_sub(head) > self.sq_mask {
                return false;
            }
            let idx = tail & self.sq_mask;
            let sqe = &mut *self.sqes.add(idx as usize);
            *sqe = IoUringSqe {
                opcode: IORING_OP_READ,
                fd,
                off: offset,
                addr: dst as u64,
                len,
                user_data,
                ..Default::default()
            };
            *self.sq_array.add(idx as usize) = idx;
            (*self.sq_tail).store(tail.wrapping_add(1), Ordering::Release);
            true
        }
    }

    pub fn pop_cqe(&self) -> Option<IoUringCqe> {
        unsafe {
            let head = (*self.cq_head).load(Ordering::Relaxed);
            let tail = (*self.cq_tail).load(Ordering::Acquire);
            if head == tail {
                return None;
            }
            let cqe = *self.cqes.add((head & self.cq_mask) as usize);
            (*self.cq_head).store(head.wrapping_add(1), Ordering::Release);
            Some(cqe)
        }
    }
}

impl Drop for IoUring {
    fn drop(&mut self) {
        unsafe {
            munmap(self.sqes as *mut c_void, self.sqes_len);
            if !self.single_mmap {
                munmap(self.cq_ptr as *mut c_void, self.cq_len);
            }
            munmap(self.sq_ptr as *mut c_void, self.sq_len);
            close(self.ring_fd);
        }
    }
}

pub struct DirectFile {
    fd: c_int,
}

impl DirectFile {
    pub fn open_readonly(path: &Path) -> Result<Self, String> {
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "invalid path".to_string())?;
        let fd = unsafe { open(cpath.as_ptr(), O_RDONLY | O_DIRECT) };
        if fd < 0 {
            return Err(format!(
                "open({}, O_RDONLY | O_DIRECT) failed: errno {}",
                path.display(),
                last_errno()
            ));
        }
        Ok(Self { fd })
    }

    pub fn fd(&self) -> c_int {
        self.fd
    }
}

impl Drop for DirectFile {
    fn drop(&mut self) {
        if self.fd >= 0 {
            unsafe { close(self.fd) };
            self.fd = -1;
        }
    }
}

pub fn bench_seq_read_odirect(
    path: &Path,
    file_bytes: usize,
    chunk_bytes: usize,
    qd: usize,
) -> Result<f64, String> {
    assert!(file_bytes.is_multiple_of(chunk_bytes));
    assert!(chunk_bytes.is_multiple_of(4096));

    let ring = IoUring::new((qd * 2) as u32)?;
    let file = DirectFile::open_readonly(path)?;
    let mut bufs = Vec::with_capacity(qd);
    for _ in 0..qd {
        bufs.push(AlignedBuf::alloc(4096, chunk_bytes)?);
    }

    let total_chunks = file_bytes / chunk_bytes;
    let mut next_submit = 0usize;
    let mut completed = 0usize;
    let mut inflight = 0usize;

    let t0 = Instant::now();
    while completed < total_chunks {
        let mut newly_pushed = 0u32;
        while inflight < qd && next_submit < total_chunks {
            let slot = next_submit % qd;
            let offset = (next_submit * chunk_bytes) as u64;
            let ok = ring.push_read(
                file.fd(),
                offset,
                bufs[slot].as_mut_ptr(),
                chunk_bytes as u32,
                slot as u64,
            );
            if !ok {
                break;
            }
            next_submit += 1;
            inflight += 1;
            newly_pushed += 1;
        }

        ring.submit_and_wait(newly_pushed, 1)?;
        while let Some(cqe) = ring.pop_cqe() {
            if cqe.res < 0 {
                return Err(format!("io_uring read CQE error: {}", cqe.res));
            }
            if cqe.res as usize != chunk_bytes {
                return Err(format!(
                    "short io_uring read: got {} expected {}",
                    cqe.res, chunk_bytes
                ));
            }
            completed += 1;
            inflight -= 1;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    Ok((file_bytes as f64) / elapsed / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn io_uring_odirect_read_1mib() {
        let path =
            std::env::temp_dir().join(format!("scoria_uring_test_{}.bin", std::process::id()));
        let len = 1024 * 1024;
        {
            let mut f = fs::File::create(&path).expect("create temp");
            let data = vec![0x5Au8; len];
            f.write_all(&data).expect("write temp");
            f.sync_all().expect("sync temp");
        }
        let res = bench_seq_read_odirect(&path, len, 256 * 1024, 4);
        let _ = fs::remove_file(&path);
        let gbps = res.expect("io_uring O_DIRECT 1MiB read");
        assert!(gbps > 0.0);
    }
}
