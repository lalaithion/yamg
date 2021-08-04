use std::sync::RwLock;

pub trait SyncedValue {
    type Inner;

    fn read_a(&self) -> Self::Inner;
    fn write_a(&self, x: Self::Inner);
}

impl<T: Clone> SyncedValue for RwLock<T> {
    type Inner = T;

    fn read_a(&self) -> Self::Inner {
        self.read().unwrap().clone()
    }

    fn write_a(&self, x: Self::Inner) {
        *self.write().unwrap() = x;
    }
}
