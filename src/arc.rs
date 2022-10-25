#![deny(missing_docs)]
use ahash::RandomState;
use std::fmt::{Debug, Display, Pointer};
type Container<T> = DashMap<BoxRefCount<T>, (), RandomState>;
type Untyped = Box<(dyn Any + Send + Sync + 'static)>;
use std::borrow::Borrow;
use std::convert::AsRef;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use dashmap::{mapref::entry::Entry, DashMap};
use std::any::Any;
use std::any::TypeId;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A pointer to a reference-counted interned object.
///
/// This type requires feature "arc".  The interned object will be held in memory only until its
/// reference count reaches zero.
///
/// # Example
/// ```rust
/// use internment::ArcIntern;
///
/// let x = ArcIntern::new("hello");
/// let y = ArcIntern::new("world");
/// assert_ne!(x, y);
/// assert_eq!(x, ArcIntern::new("hello"));
/// assert_eq!(*x, "hello"); // dereference an ArcIntern like a pointer
/// ```
///
/// # Example with owned `String` data
///
/// ```rust
/// use internment::ArcIntern;
///
/// let x = ArcIntern::new("hello".to_string());
/// let y = ArcIntern::<String>::from_ref("world");
/// assert_eq!(x, ArcIntern::from_ref("hello"));
/// assert_eq!(&*x, "hello"); // dereference an ArcIntern like a pointer
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "arc")))]
pub struct ArcIntern<T: ?Sized + Eq + Hash + Send + Sync + 'static> {
    pointer: std::ptr::NonNull<RefCount<T>>,
}

unsafe impl<T: ?Sized + Eq + Hash + Send + Sync> Send for ArcIntern<T> {}
unsafe impl<T: ?Sized + Eq + Hash + Send + Sync> Sync for ArcIntern<T> {}

#[derive(Debug)]
struct RefCount<T: ?Sized> {
    count: AtomicUsize,
    data: T,
}

#[cfg(feature = "dst")]
impl<T: Copy> RefCount<[T]> {
    fn from_slice(slice: &[T]) -> Box<RefCount<[T]>> {
        use std::alloc::{Allocator, Layout};
        let layout = Layout::new::<RefCount<()>>()
            .extend(Layout::array::<T>(slice.len()).unwrap())
            .unwrap()
            .0
            .pad_to_align();
        let ptr = std::alloc::Global.allocate_zeroed(layout).unwrap();
        let ptr = std::ptr::slice_from_raw_parts_mut(ptr.as_ptr() as *mut T, slice.len())
            as *mut RefCount<[T]>;
        let mut this = unsafe { Box::from_raw(ptr) };

        this.count = AtomicUsize::new(1);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), &mut this.data[0] as *mut T, slice.len())
        };
        this
    }
}
#[cfg(feature = "dst")]
impl RefCount<str> {
    fn from_str(s: &str) -> Box<RefCount<str>> {
        let bytes = s.as_bytes();
        let boxed_refcount = RefCount::<[u8]>::from_slice(bytes);
        debug_assert_eq!(s.len(), boxed_refcount.data.len());

        unsafe { Box::from_raw(Box::into_raw(boxed_refcount) as *mut RefCount<str>) }
    }
}

impl<T: ?Sized + Eq> Eq for RefCount<T> {}
impl<T: ?Sized + PartialEq> PartialEq for RefCount<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl<T: ?Sized + Hash> Hash for RefCount<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.data.hash(hasher)
    }
}

#[derive(Eq, PartialEq, Hash)]
struct BoxRefCount<T: ?Sized>(Box<RefCount<T>>);
impl<T> BoxRefCount<T> {
    fn into_inner(self) -> T {
        self.0.data
    }
}

impl<T: ?Sized> Borrow<T> for BoxRefCount<T> {
    fn borrow(&self) -> &T {
        &self.0.data
    }
}
impl<T: ?Sized> Borrow<RefCount<T>> for BoxRefCount<T> {
    fn borrow(&self) -> &RefCount<T> {
        &self.0
    }
}
impl<T: ?Sized> Deref for BoxRefCount<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0.data
    }
}

impl<T: ?Sized + Eq + Hash + Send + Sync + 'static> ArcIntern<T> {
    fn get_pointer(&self) -> *const RefCount<T> {
        self.pointer.as_ptr()
    }
    fn get_container() -> dashmap::mapref::one::Ref<'static, TypeId, Untyped, RandomState> {
        use once_cell::sync::OnceCell;
        static ARC_CONTAINERS: OnceCell<DashMap<TypeId, Untyped, RandomState>> = OnceCell::new();
        let type_map = ARC_CONTAINERS.get_or_init(|| DashMap::with_hasher(RandomState::new()));
        // Prefer taking the read lock to reduce contention, only use entry api if necessary.
        let boxed = if let Some(boxed) = type_map.get(&TypeId::of::<T>()) {
            boxed
        } else {
            type_map
                .entry(TypeId::of::<T>())
                .or_insert_with(|| Box::new(Container::<T>::with_hasher(RandomState::new())))
                .downgrade()
        };
        boxed
    }

    /// Intern a value from a reference with atomic reference counting.
    ///
    /// If this value has not previously been
    /// interned, then `new` will allocate a spot for the value on the
    /// heap and generate that value using `T::from(val)`.
    pub fn from_ref<'a, Q: ?Sized + Eq + Hash + 'a>(val: &'a Q) -> ArcIntern<T>
    where
        T: Borrow<Q> + From<&'a Q>,
    {
        // No reference only fast-path as
        // the trait `std::borrow::Borrow<Q>` is not implemented for `Arc<T>`
        Self::new(val.into())
    }
    /// See how many objects have been interned.  This may be helpful
    /// in analyzing memory use.
    pub fn num_objects_interned() -> usize {
        let c = Self::get_container();
        c.downcast_ref::<Container<T>>()
            .map(|m| m.len())
            .unwrap_or(0)
    }
    /// Return the number of counts for this pointer.
    pub fn refcount(&self) -> usize {
        unsafe { self.pointer.as_ref().count.load(Ordering::Acquire) }
    }

    /// Only for benchmarking, this will cause problems
    #[cfg(feature = "bench")]
    pub fn benchmarking_only_clear_interns() {}

    /// make new [`ArcIntern`] with copyable initial value, like `&str` or `&[u8]`.
    fn new_with_copyable_init_val<I, NewFn>(val: &I, new_fn: NewFn) -> ArcIntern<T>
    where
        I: ?Sized + Hash + std::cmp::Eq,
        BoxRefCount<T>: Borrow<I>,
        NewFn: Fn(&I) -> Box<RefCount<T>>,
    {
        // cache the converted BoxRefCount
        let mut converted = None;
        loop {
            let c = Self::get_container();
            let m = c.downcast_ref::<Container<T>>().unwrap();

            if let Some(b) = m.get_mut(val) {
                let b = b.key();
                // First increment the count.  We are holding the write mutex here.
                // Has to be the write mutex to avoid a race
                let oldval = b.0.count.fetch_add(1, Ordering::SeqCst);
                if oldval != 0 {
                    // we can only use this value if the value is not about to be freed
                    return ArcIntern {
                        pointer: std::ptr::NonNull::from(b.0.borrow()),
                    };
                } else {
                    // we have encountered a race condition here.
                    // we will just wait for the object to finish
                    // being freed.
                    b.0.count.fetch_sub(1, Ordering::SeqCst);
                }
            } else {
                let b = std::mem::take(&mut converted).unwrap_or_else(|| new_fn(val));
                match m.entry(BoxRefCount(b)) {
                    Entry::Vacant(e) => {
                        // We can insert, all is good
                        let p = ArcIntern {
                            pointer: std::ptr::NonNull::from(e.key().0.borrow()),
                        };
                        e.insert(());
                        return p;
                    }
                    Entry::Occupied(e) => {
                        // Race, map already has data, go round again
                        let box_ref_count = e.into_key();
                        converted = Some(box_ref_count.0);
                    }
                }
            }
            // yield so that the object can finish being freed,
            // and then we will be able to intern a new copy.
            std::thread::yield_now();
        }
    }
}

impl<T: Eq + Hash + Send + Sync + 'static> ArcIntern<T> {
    /// Intern a value.  If this value has not previously been
    /// interned, then `new` will allocate a spot for the value on the
    /// heap.  Otherwise, it will return a pointer to the object
    /// previously allocated.
    ///
    /// Note that `ArcIntern::new` is a bit slow, since it needs to check
    /// a `DashMap` which is protected by internal sharded locks.
    pub fn new(mut val: T) -> ArcIntern<T> {
        loop {
            let c = Self::get_container();
            let m = c.downcast_ref::<Container<T>>().unwrap();
            if let Some(b) = m.get_mut(&val) {
                let b = b.key();
                // First increment the count.  We are holding the write mutex here.
                // Has to be the write mutex to avoid a race
                let oldval = b.0.count.fetch_add(1, Ordering::SeqCst);
                if oldval != 0 {
                    // we can only use this value if the value is not about to be freed
                    return ArcIntern {
                        pointer: std::ptr::NonNull::from(b.0.borrow()),
                    };
                } else {
                    // we have encountered a race condition here.
                    // we will just wait for the object to finish
                    // being freed.
                    b.0.count.fetch_sub(1, Ordering::SeqCst);
                }
            } else {
                let b = Box::new(RefCount {
                    count: AtomicUsize::new(1),
                    data: val,
                });
                match m.entry(BoxRefCount(b)) {
                    Entry::Vacant(e) => {
                        // We can insert, all is good
                        let p = ArcIntern {
                            pointer: std::ptr::NonNull::from(e.key().0.borrow()),
                        };
                        e.insert(());
                        return p;
                    }
                    Entry::Occupied(e) => {
                        // Race, map already has data, go round again
                        let box_ref_count = e.into_key();
                        val = box_ref_count.into_inner();
                    }
                }
            }
            // yield so that the object can finish being freed,
            // and then we will be able to intern a new copy.
            std::thread::yield_now();
        }
    }
}

impl<T: Eq + Hash + Send + Sync + 'static> Clone for ArcIntern<T> {
    fn clone(&self) -> Self {
        // First increment the count.  Using a relaxed ordering is
        // alright here, as knowledge of the original reference
        // prevents other threads from erroneously deleting the
        // object.  (See `std::sync::Arc` documentation for more
        // explanation.)
        unsafe { self.pointer.as_ref().count.fetch_add(1, Ordering::Relaxed) };
        ArcIntern {
            pointer: self.pointer,
        }
    }
}

#[cfg(not(test))]
fn yield_on_tests() {}
#[cfg(test)]
fn yield_on_tests() {
    std::thread::yield_now();
}

impl<T: ?Sized + Eq + Hash + Send + Sync> Drop for ArcIntern<T> {
    fn drop(&mut self) {
        // (Quoting from std::sync::Arc again): Because `fetch_sub` is
        // already atomic, we do not need to synchronize with other
        // threads unless we are going to delete the object. This same
        // logic applies to the below `fetch_sub` to the `weak` count.
        let count_was = unsafe { self.pointer.as_ref().count.fetch_sub(1, Ordering::SeqCst) };
        if count_was == 1 {
            // The following causes the code only when testing, to yield
            // control before taking the mutex, which should make it
            // easier to trigger any race condition (and hopefully won't
            // mask any other race conditions).
            yield_on_tests();
            // (Quoting from std::sync::Arc again): This fence is
            // needed to prevent reordering of use of the data and
            // deletion of the data.  Because it is marked `Release`,
            // the decreasing of the reference count synchronizes with
            // this `Acquire` fence. This means that use of the data
            // happens before decreasing the reference count, which
            // happens before this fence, which happens before the
            // deletion of the data.
            std::sync::atomic::fence(Ordering::SeqCst);

            // removed is declared before m, so the mutex guard will be
            // dropped *before* the removed content is dropped, since it
            // might need to lock the mutex.
            let _remove;
            let c = Self::get_container();
            let m = c.downcast_ref::<Container<T>>().unwrap();
            _remove = m.remove(unsafe { self.pointer.as_ref() });
        }
    }
}

impl<T: ?Sized + Send + Sync + Hash + Eq> AsRef<T> for ArcIntern<T> {
    fn as_ref(&self) -> &T {
        unsafe { &self.pointer.as_ref().data }
    }
}

impl<T: ?Sized + Eq + Hash + Send + Sync> Deref for ArcIntern<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.as_ref()
    }
}

impl<T: ?Sized + Eq + Hash + Send + Sync + Display> Display for ArcIntern<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.deref().fmt(f)
    }
}

impl<T: ?Sized + Eq + Hash + Send + Sync> Pointer for ArcIntern<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        Pointer::fmt(&self.get_pointer(), f)
    }
}

/// The hash implementation returns the hash of the pointer
/// value, not the hash of the value pointed to.  This should
/// be irrelevant, since there is a unique pointer for every
/// value, but it *is* observable, since you could compare the
/// hash of the pointer with hash of the data itself.
impl<T: ?Sized + Eq + Hash + Send + Sync> Hash for ArcIntern<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get_pointer().hash(state);
    }
}

impl<T: ?Sized + Eq + Hash + Send + Sync> PartialEq for ArcIntern<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get_pointer() == other.get_pointer()
    }
}
impl<T: ?Sized + Eq + Hash + Send + Sync> Eq for ArcIntern<T> {}

impl<T: ?Sized + Eq + Hash + Send + Sync + PartialOrd> PartialOrd for ArcIntern<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.as_ref().partial_cmp(other)
    }
    fn lt(&self, other: &Self) -> bool {
        self.as_ref().lt(other)
    }
    fn le(&self, other: &Self) -> bool {
        self.as_ref().le(other)
    }
    fn gt(&self, other: &Self) -> bool {
        self.as_ref().gt(other)
    }
    fn ge(&self, other: &Self) -> bool {
        self.as_ref().ge(other)
    }
}
impl<T: ?Sized + Eq + Hash + Send + Sync + Ord> Ord for ArcIntern<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_ref().cmp(other)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
#[cfg(feature = "serde")]
impl<T: ?Sized + Eq + Hash + Send + Sync + Serialize> Serialize for ArcIntern<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(serializer)
    }
}

impl<T: Eq + Hash + Send + Sync + 'static> From<T> for ArcIntern<T> {
    fn from(t: T) -> Self {
        ArcIntern::new(t)
    }
}

impl<T: Eq + Hash + Send + Sync + Default + 'static> Default for ArcIntern<T> {
    fn default() -> Self {
        ArcIntern::new(Default::default())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
#[cfg(feature = "serde")]
impl<'de, T> Deserialize<'de> for ArcIntern<T>
where
    T: Eq + Hash + Send + Sync + 'static + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(|x: T| Self::new(x))
    }
}

#[cfg(feature = "dst")]
mod dst {
    use super::*;
    impl From<&str> for ArcIntern<str> {
        fn from(s: &str) -> Self {
            ArcIntern::<str>::new_with_copyable_init_val(s, |s| RefCount::<str>::from_str(s))
        }
    }
    impl From<String> for ArcIntern<str> {
        fn from(s: String) -> Self {
            Self::from(&s[..])
        }
    }
    impl From<Box<str>> for ArcIntern<str> {
        fn from(s: Box<str>) -> Self {
            Self::from(&s[..])
        }
    }
    impl Default for ArcIntern<str> {
        fn default() -> Self {
            Self::from("")
        }
    }

    impl<T> From<&[T]> for ArcIntern<[T]>
    where
        T: Copy + Send + Sync + Hash + Eq + 'static,
    {
        fn from(slice: &[T]) -> Self {
            ArcIntern::<[T]>::new_with_copyable_init_val(slice, |slice| {
                RefCount::<[T]>::from_slice(slice)
            })
        }
    }
    // NOTE: we can do better by removing the `Copy` bound and
    // copying the data without dropping `T`. See [`std::sync::Arc::from_box`]
    impl<T> From<Vec<T>> for ArcIntern<[T]>
    where
        T: Copy + Send + Sync + Hash + Eq + 'static,
    {
        fn from(vec: Vec<T>) -> Self {
            Self::from(&vec[..])
        }
    }
    // NOTE: we can do better by removing the `Copy` bound and
    // copying the data without dropping `T`. See [`std::sync::Arc::from_box`]
    impl<T> From<Box<[T]>> for ArcIntern<[T]>
    where
        T: Copy + Send + Sync + Hash + Eq + 'static,
    {
        fn from(boxed_slice: Box<[T]>) -> Self {
            Self::from(&boxed_slice[..])
        }
    }
    impl<T> Default for ArcIntern<[T]>
    where
        T: Copy + Send + Sync + Hash + Eq + 'static,
    {
        fn default() -> Self {
            Self::from(&[][..])
        }
    }

    // implement some useful equal comparisons
    macro_rules! impl_eq {
        ([$($vars:tt)*] $lhs:ty, $rhs: ty) => {
            #[allow(unused_lifetimes)]
            impl<'a, $($vars)*> PartialEq<$rhs> for $lhs {
                #[inline]
                fn eq(&self, other: &$rhs) -> bool {
                    PartialEq::eq(&self[..], &other[..])
                }
                #[inline]
                fn ne(&self, other: &$rhs) -> bool {
                    PartialEq::ne(&self[..], &other[..])
                }
            }

            #[allow(unused_lifetimes)]
            impl<'a, $($vars)*> PartialEq<$lhs> for $rhs {
                #[inline]
                fn eq(&self, other: &$lhs) -> bool {
                    PartialEq::eq(&self[..], &other[..])
                }
                #[inline]
                fn ne(&self, other: &$lhs) -> bool {
                    PartialEq::ne(&self[..], &other[..])
                }
            }
        };
    }
    impl_eq! { [] ArcIntern<str>, str }
    impl_eq! { [] ArcIntern<str>, &'a str }
    impl_eq! { [] ArcIntern<str>, String }
    impl_eq! { [] ArcIntern<str>, std::borrow::Cow<'a, str> }
    impl_eq! { [] ArcIntern<str>, Box<str> }
    impl_eq! { [] ArcIntern<str>, std::rc::Rc<str> }
    impl_eq! { [] ArcIntern<str>, std::sync::Arc<str> }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, Vec<T> }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, [T] }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, &'a [T] }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, &'a mut [T] }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, std::borrow::Cow<'a, [T]> }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, Box<[T]> }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, std::rc::Rc<[T]> }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static] ArcIntern<[T]>, std::sync::Arc<[T]> }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static, const N: usize] ArcIntern<[T]>, [T; N] }
    impl_eq! { [T: Copy + Send + Sync + Hash + Eq + 'static, const N: usize] ArcIntern<[T]>, &[T; N] }

    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[cfg(feature = "serde")]
    impl<'de: 'a, 'a> Deserialize<'de> for ArcIntern<str> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let s: &'a str = <&str as Deserialize>::deserialize(deserializer)?;
            Ok(Self::from(s))
        }
    }
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[cfg(feature = "serde")]
    impl<'de: 'a, 'a> Deserialize<'de> for ArcIntern<[u8]> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let slice = <&'a [u8] as Deserialize>::deserialize(deserializer)?;
            Ok(Self::from(slice))
        }
    }

    #[test]
    fn dst_arc_intern_is_sized() {
        struct _Assure
        where
            ArcIntern<str>: Sized;
        struct _Assure2
        where
            ArcIntern<[u8]>: Sized;
    }

    #[test]
    fn dst_arc_intern_is_hash() {
        struct _Assure
        where
            ArcIntern<str>: Hash;
    }

    #[test]
    fn dst_arc_intern_is_send_and_sync() {
        struct _Assure
        where
            ArcIntern<str>: Send + Sync;
    }

    #[test]
    fn common_equal_comparisons() {
        let s1: ArcIntern<str> = ArcIntern::from("hello");
        let s2: &str = "hello";
        assert_eq!(s1, s2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_arc_intern_str() {
        let s = "\"a\"";
        let mut deserializer = serde_json::Deserializer::from_str(s);
        let s = <ArcIntern<str> as serde::Deserialize>::deserialize(&mut deserializer).unwrap();
        assert_eq!(s, "a");
        assert_eq!("a", s);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialize_arc_intern_str() {
        let s = ArcIntern::<str>::from("a");
        let s = serde_json::to_string(&s).unwrap();
        assert_eq!(s, "\"a\"");
    }

    #[test]
    fn arc_intern_str() {
        let x: ArcIntern<str> = ArcIntern::from("hello");
        assert_eq!(x.len(), 5);
        assert_eq!(x.refcount(), 1);

        let y: ArcIntern<str> = ArcIntern::from("hello");
        assert_eq!(x.refcount(), 2);
        assert_eq!(y.refcount(), 2);

        assert_eq!(x.as_ptr(), y.as_ptr());
        assert_eq!(x, y);

        let z: ArcIntern<str> = ArcIntern::from(String::from("hello"));
        assert_eq!(x.refcount(), 3);
        assert_eq!(y.refcount(), 3);
        assert_eq!(z.refcount(), 3);
    }

    #[test]
    fn zst_for_dst() {
        let vec = vec![(); 500];
        let x: ArcIntern<[()]> = ArcIntern::from(vec.clone());
        assert_eq!(x.len(), 500);
        assert_eq!(x.refcount(), 1);

        let y: ArcIntern<[()]> = ArcIntern::from(vec);
        assert_eq!(x.refcount(), 2);
        assert_eq!(y.refcount(), 2);

        assert_eq!(x, y);
    }
}

#[cfg(test)]
mod arc_test {
    use super::ArcIntern;
    use super::{Borrow, Deref};
    #[test]
    fn eq_string() {
        assert_eq!(ArcIntern::new("hello"), ArcIntern::new("hello"));
        assert_ne!(ArcIntern::new("goodbye"), ArcIntern::new("farewell"));
    }
    #[test]
    fn display() {
        let world = ArcIntern::new("world");
        println!("Hello {}", world);
    }
    #[test]
    fn debug() {
        let world = ArcIntern::new("world");
        println!("Hello {:?}", world);
    }
    #[test]
    fn has_default() {
        assert_eq!(
            ArcIntern::<Option<String>>::default(),
            ArcIntern::<Option<String>>::new(None)
        );
    }
    #[test]
    fn can_clone() {
        assert_eq!(
            ArcIntern::<Option<String>>::default().clone(),
            ArcIntern::<Option<String>>::new(None)
        );
    }
    #[test]
    fn has_borrow() {
        let x = ArcIntern::<Option<String>>::default();
        let b: &Option<String> = x.borrow();
        assert_eq!(b, ArcIntern::<Option<String>>::new(None).as_ref());
    }
    #[test]
    fn has_deref() {
        let x = ArcIntern::<Option<String>>::default();
        let b: &Option<String> = x.as_ref();
        assert_eq!(b, ArcIntern::<Option<String>>::new(None).deref());
    }
}

#[test]
fn test_arcintern_freeing() {
    assert_eq!(ArcIntern::<i32>::num_objects_interned(), 0);
    assert_eq!(ArcIntern::new(5), ArcIntern::new(5));
    {
        let _interned = ArcIntern::new(6);
        assert_eq!(ArcIntern::<i32>::num_objects_interned(), 1);
    }
    {
        let _interned = ArcIntern::new(6);
        assert_eq!(ArcIntern::<i32>::num_objects_interned(), 1);
    }
    {
        let _interned = ArcIntern::new(7);
        assert_eq!(ArcIntern::<i32>::num_objects_interned(), 1);
    }

    let six = ArcIntern::new(6);

    {
        let _interned = ArcIntern::new(7);
        assert_eq!(ArcIntern::<i32>::num_objects_interned(), 2);
    }
    assert_eq!(ArcIntern::new(6), six);
}

#[test]
fn test_arcintern_nested_drop() {
    #[derive(PartialEq, Eq, Hash)]
    enum Nat {
        Zero,
        Successor(ArcIntern<Nat>),
    }
    let zero = ArcIntern::new(Nat::Zero);
    let _one = ArcIntern::new(Nat::Successor(zero));
}

impl<T: ?Sized + Eq + Hash + Send + Sync + Debug> Debug for ArcIntern<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        Pointer::fmt(&self.pointer, f)?;
        f.write_str(" : ")?;
        self.deref().fmt(f)
    }
}

#[cfg(test)]
#[derive(Eq, PartialEq, Hash)]
pub struct TestStructCount(String, u64, std::sync::Arc<bool>);

#[cfg(test)]
#[derive(Eq, PartialEq, Hash)]
pub struct TestStruct(String, u64);

// Quickly create and destroy a small number of interned objects from
// multiple threads.
#[test]
fn multithreading1() {
    use std::sync::Arc;
    use std::thread;
    let mut thandles = vec![];
    let drop_check = Arc::new(true);
    for _i in 0..10 {
        thandles.push(thread::spawn({
            let drop_check = drop_check.clone();
            move || {
                for _i in 0..100_000 {
                    let _interned1 =
                        ArcIntern::new(TestStructCount("foo".to_string(), 5, drop_check.clone()));
                    let _interned2 =
                        ArcIntern::new(TestStructCount("bar".to_string(), 10, drop_check.clone()));
                }
            }
        }));
    }
    for h in thandles.into_iter() {
        h.join().unwrap()
    }
    assert_eq!(Arc::strong_count(&drop_check), 1);
    assert_eq!(ArcIntern::<TestStructCount>::num_objects_interned(), 0);
}

#[test]
fn arc_has_niche() {
    assert_eq!(
        std::mem::size_of::<ArcIntern<String>>(),
        std::mem::size_of::<usize>(),
    );
    assert_eq!(
        std::mem::size_of::<Option<ArcIntern<String>>>(),
        std::mem::size_of::<usize>(),
    );
}

#[test]
fn like_doctest_arcintern() {
    let x = ArcIntern::new("hello".to_string());
    let y = ArcIntern::<String>::from_ref("world");
    assert_ne!(x, y);
    assert_eq!(x, ArcIntern::from_ref("hello"));
    assert_eq!(y, ArcIntern::from_ref("world"));
    assert_eq!(&*x, "hello"); // dereference a Intern like a pointer\
}
