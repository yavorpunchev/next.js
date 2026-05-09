//! A `Vec`-shaped container with `u8` length and capacity, sized 16 B on 64-bit instead of 24 B.
//!
//! Used by `#[task_storage]` for `TaskStorage`'s lazy-fields field, which holds at most ~25
//! elements (one per declared lazy field in the schema). With several million task storages
//! live during a typical Next.js build, the 8 B saved per task adds up to dozens of MB of
//! resident memory.
//!
//! The API is intentionally a strict subset of `Vec` covering only what the task-storage
//! callers and the `#[task_storage]` macro emit need: `len`, `is_empty`, `iter`, `iter_mut`,
//! `push`, `swap_remove`, `last_mut`, `index`, `index_mut`, `extend`, `reserve`, `Default`,
//! `Debug`, `ShrinkToFit`. No `Clone` or `PartialEq` — `TaskStorage` doesn't derive them.
//!
//! Capacity is bounded by `u8::MAX = 255`. The schema currently uses ~25 variants and growth
//! follows powers of two, so we have plenty of headroom; oversized pushes panic.

use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    fmt,
    iter::FromIterator,
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr::{self, NonNull},
};

/// Compact `Vec`-shaped container; see module docs for rationale.
pub struct LazyVec<T> {
    /// Heap pointer. Dangling (uninitialized) when `cap == 0`.
    ptr: NonNull<T>,
    len: u8,
    cap: u8,
    /// Marker so we own `T` for drop-check purposes (matches `Vec<T>`'s variance/dropck).
    _marker: PhantomData<T>,
}

// SAFETY: same as `Vec<T>` — we own a heap allocation of `T`s, and the only shared state is via
// the `ptr` which is unique to this `LazyVec`.
unsafe impl<T: Send> Send for LazyVec<T> {}
unsafe impl<T: Sync> Sync for LazyVec<T> {}

impl<T> Default for LazyVec<T> {
    fn default() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            _marker: PhantomData,
        }
    }
}

impl<T> LazyVec<T> {
    pub const fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap as usize
    }

    /// Returns an iterator over the elements in insertion order.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the elements in insertion order.
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is valid for `len` initialized elements; if len == 0, slicing the
        // dangling pointer is allowed by `from_raw_parts`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same as `as_slice`; we hold `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    /// Appends `value`. Panics if growing the buffer would exceed `u8::MAX` capacity.
    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.grow_by_one();
        }
        // SAFETY: `len < cap` after the grow; the slot at index `len` is uninitialized and we
        // initialize it here.
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len()), value);
        }
        self.len += 1;
    }

    /// Removes the element at `idx` by swapping it with the last and popping. O(1).
    /// Panics if `idx` is out of bounds (matching `Vec::swap_remove`).
    pub fn swap_remove(&mut self, idx: usize) -> T {
        let len = self.len();
        assert!(idx < len, "swap_remove index out of bounds: {idx} >= {len}");
        // SAFETY: `idx < len`; we read out the value and then either swap or shrink.
        unsafe {
            let last = self.ptr.as_ptr().add(len - 1);
            let hole = self.ptr.as_ptr().add(idx);
            let value = ptr::read(hole);
            if idx != len - 1 {
                ptr::copy_nonoverlapping(last, hole, 1);
            }
            self.len -= 1;
            value
        }
    }

    /// Returns a mutable reference to the last element if any.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.as_mut_slice().last_mut()
    }

    /// Reserves capacity for at least `additional` more elements. No-op if already sufficient.
    /// Panics if the resulting capacity would exceed `u8::MAX`.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len() + additional;
        if needed <= self.cap as usize {
            return;
        }
        let new_cap = needed.next_power_of_two().max(4);
        self.realloc_to(new_cap);
    }

    /// Grow the buffer by at least one slot. The first allocation jumps to 4 to amortize the
    /// initial pushes; subsequent growths double up to the `u8::MAX` ceiling.
    #[cold]
    #[inline(never)]
    fn grow_by_one(&mut self) {
        let new_cap = if self.cap == 0 {
            4
        } else {
            (self.cap as usize) * 2
        };
        self.realloc_to(new_cap);
    }

    fn realloc_to(&mut self, new_cap: usize) {
        assert!(
            new_cap <= u8::MAX as usize,
            "LazyVec capacity overflow: requested {new_cap}, max {}",
            u8::MAX
        );
        if new_cap == self.cap as usize {
            return;
        }
        if size_of::<T>() == 0 {
            // Zero-sized types: no allocation needed; just bump cap.
            self.cap = new_cap as u8;
            return;
        }

        // Allocate new buffer.
        let new_layout = Layout::array::<T>(new_cap).expect("LazyVec layout overflow");
        // SAFETY: Layout has nonzero size because new_cap > 0 (or we'd not be here) and T is
        // nonzero-sized (handled above).
        let new_ptr = unsafe { alloc(new_layout) } as *mut T;
        let new_ptr = match NonNull::new(new_ptr) {
            Some(p) => p,
            None => handle_alloc_error(new_layout),
        };

        // Move elements over.
        if self.cap > 0 {
            // SAFETY: old buffer holds `len` initialized Ts; copy them to the new buffer's
            // prefix (which is uninitialized).
            unsafe {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len());
            }
            self.deallocate_old();
        }

        self.ptr = new_ptr;
        self.cap = new_cap as u8;
    }

    /// Deallocates the current heap buffer without dropping the elements (caller must have
    /// already moved or dropped them). No-op if `cap == 0`.
    fn deallocate_old(&mut self) {
        if self.cap == 0 || size_of::<T>() == 0 {
            return;
        }
        let old_layout =
            Layout::array::<T>(self.cap as usize).expect("LazyVec layout was valid when allocated");
        // SAFETY: ptr came from `alloc` with this layout in `realloc_to`.
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
        }
    }

    /// Shrinks the heap buffer to fit `len`, freeing it entirely if `len == 0`.
    pub fn shrink_to_fit(&mut self) {
        if (self.len as usize) == (self.cap as usize) {
            return;
        }
        if self.len == 0 {
            // Free the buffer entirely.
            self.deallocate_old();
            self.ptr = NonNull::dangling();
            self.cap = 0;
            return;
        }
        let new_cap = self.len as usize;
        // Allocate a smaller buffer, copy, free old.
        let new_layout = Layout::array::<T>(new_cap).expect("LazyVec layout overflow");
        // SAFETY: layout is nonzero (new_cap > 0, T is nonzero-sized — ZST early-returned via the
        // len == cap check above since cap = 0 for ZSTs would also trigger the equal branch).
        let new_ptr = unsafe { alloc(new_layout) } as *mut T;
        let new_ptr = match NonNull::new(new_ptr) {
            Some(p) => p,
            None => handle_alloc_error(new_layout),
        };
        // SAFETY: old buffer holds `len` initialized Ts.
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len());
        }
        self.deallocate_old();
        self.ptr = new_ptr;
        self.cap = new_cap as u8;
    }
}

impl<T> std::ops::Index<usize> for LazyVec<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.as_slice()[idx]
    }
}

impl<T> std::ops::IndexMut<usize> for LazyVec<T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.as_mut_slice()[idx]
    }
}

impl<T> std::ops::Deref for LazyVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for LazyVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Extend<T> for LazyVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lo, _) = iter.size_hint();
        if lo > 0 {
            self.reserve(lo);
        }
        for item in iter {
            self.push(item);
        }
    }
}

impl<T> FromIterator<T> for LazyVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        v.extend(iter);
        v
    }
}

impl<T> IntoIterator for LazyVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        // Move out of self without running its Drop (we'll drop unmoved elements ourselves).
        let me = ManuallyDrop::new(self);
        let ptr = me.ptr;
        let len = me.len;
        let cap = me.cap;
        IntoIter {
            buf: ptr,
            cap,
            start: ptr,
            // SAFETY: end = ptr + len, where len <= cap. Valid one-past-the-end pointer.
            end: unsafe { ptr.as_ptr().add(len as usize) },
            _marker: PhantomData,
        }
    }
}

impl<'a, T> IntoIterator for &'a LazyVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut LazyVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> std::slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

/// Owning iterator returned by [`LazyVec::into_iter`].
pub struct IntoIter<T> {
    /// The underlying buffer (kept so we can deallocate in Drop).
    buf: NonNull<T>,
    cap: u8,
    /// Pointer to the next element to yield.
    start: NonNull<T>,
    /// One-past-the-end pointer. Equal to `start` when exhausted.
    end: *mut T,
    _marker: PhantomData<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start.as_ptr() == self.end {
            None
        } else {
            // SAFETY: start points to an initialized element. We bump start past it after read.
            unsafe {
                let v = ptr::read(self.start.as_ptr());
                self.start = NonNull::new_unchecked(self.start.as_ptr().add(1));
                Some(v)
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        // SAFETY: end >= start by construction.
        let n = unsafe { self.end.offset_from(self.start.as_ptr()) } as usize;
        (n, Some(n))
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // Drop any remaining elements.
        while self.next().is_some() {}
        // Deallocate the buffer.
        if self.cap > 0 && size_of::<T>() != 0 {
            let layout = Layout::array::<T>(self.cap as usize)
                .expect("LazyVec layout was valid when allocated");
            // SAFETY: buf was allocated via `alloc` with this layout.
            unsafe {
                dealloc(self.buf.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for LazyVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Drop for LazyVec<T> {
    fn drop(&mut self) {
        // Drop populated elements in place.
        if self.len > 0 {
            // SAFETY: we own `len` initialized elements at the start of the buffer.
            unsafe {
                ptr::drop_in_place(std::slice::from_raw_parts_mut(
                    self.ptr.as_ptr(),
                    self.len(),
                ));
            }
        }
        self.deallocate_old();
    }
}

impl<T> shrink_to_fit::ShrinkToFit for LazyVec<T> {
    fn shrink_to_fit(&mut self) {
        Self::shrink_to_fit(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size() {
        // The whole point: 16 B on 64-bit, vs 24 B for Vec.
        assert_eq!(std::mem::size_of::<LazyVec<u64>>(), 16);
        assert_eq!(std::mem::size_of::<LazyVec<[u8; 48]>>(), 16);
    }

    #[test]
    fn push_iter_swap_remove() {
        let mut v: LazyVec<u32> = LazyVec::new();
        assert!(v.is_empty());
        v.push(10);
        v.push(20);
        v.push(30);
        assert_eq!(v.len(), 3);
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![10, 20, 30]);
        let removed = v.swap_remove(0);
        assert_eq!(removed, 10);
        // After swap_remove(0), buffer is [30, 20] (last swapped into hole).
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![30, 20]);
        assert_eq!(v[0], 30);
        assert_eq!(v[1], 20);
    }

    #[test]
    fn growth_pattern() {
        let mut v: LazyVec<u32> = LazyVec::new();
        for i in 0..32u32 {
            v.push(i);
        }
        assert_eq!(v.len(), 32);
        let collected: Vec<u32> = v.iter().copied().collect();
        assert_eq!(collected, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn extend_and_reserve() {
        let mut v: LazyVec<u32> = LazyVec::new();
        v.extend(0..10);
        assert_eq!(v.len(), 10);
        v.reserve(5);
        assert!(v.capacity() >= 15);
    }

    #[test]
    fn last_mut_and_index_mut() {
        let mut v: LazyVec<u32> = LazyVec::new();
        v.push(1);
        v.push(2);
        *v.last_mut().unwrap() = 99;
        assert_eq!(v[1], 99);
        v[0] = 7;
        assert_eq!(v[0], 7);
    }

    #[test]
    fn drop_runs_on_elements() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct DropCounter<'a>(&'a AtomicUsize);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let count = AtomicUsize::new(0);
        {
            let mut v: LazyVec<DropCounter<'_>> = LazyVec::new();
            v.push(DropCounter(&count));
            v.push(DropCounter(&count));
            v.push(DropCounter(&count));
        }
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn into_iter_drops_and_yields() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct DropCounter<'a>(&'a AtomicUsize, u32);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let count = AtomicUsize::new(0);
        let mut v: LazyVec<DropCounter<'_>> = LazyVec::new();
        v.push(DropCounter(&count, 1));
        v.push(DropCounter(&count, 2));
        v.push(DropCounter(&count, 3));

        let mut iter = v.into_iter();
        assert_eq!(iter.next().unwrap().1, 1);
        assert_eq!(iter.next().unwrap().1, 2);
        // Drop iterator with one remaining element.
        drop(iter);
        // 3 total drops: the two yielded + the one remaining in the iter.
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn shrink_to_fit_releases_buffer() {
        let mut v: LazyVec<u32> = LazyVec::new();
        v.extend(0..10);
        assert!(v.capacity() >= 10);
        for _ in 0..10 {
            v.swap_remove(0);
        }
        assert!(v.is_empty());
        v.shrink_to_fit();
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    #[should_panic(expected = "LazyVec capacity overflow")]
    fn capacity_overflow_panics() {
        let mut v: LazyVec<u8> = LazyVec::new();
        for _ in 0..255u32 {
            v.push(0);
        }
        // The 256th push triggers grow_by_one with new_cap = 256, which exceeds u8::MAX.
        v.push(0);
    }
}
