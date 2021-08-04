use rayon::prelude::*;
use std::iter::*;
use std::ops::{Index, IndexMut};

pub fn to_signed_index(x: (usize, usize)) -> (i64, i64) {
    (x.0 as i64, x.1 as i64)
}

pub fn left(x: (i64, i64)) -> (i64, i64) {
    (x.0 - 1, x.1)
}

pub fn right(x: (i64, i64)) -> (i64, i64) {
    (x.0 + 1, x.1)
}

pub fn up(x: (i64, i64)) -> (i64, i64) {
    (x.0, x.1 - 1)
}

pub fn down(x: (i64, i64)) -> (i64, i64) {
    (x.0, x.1 + 1)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Grid<T> {
    stride: usize,
    data: Vec<T>,
}

pub static NEIGHBOR_OFFSETS: [(i64, i64); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

pub static ORTHOGONAL_OFFSETS: [(i64, i64); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

pub fn shift_index(index: (usize, usize), offset: (i64, i64)) -> (i64, i64) {
    (index.0 as i64 + offset.0, index.1 as i64 + offset.1)
}

pub fn add_offsets(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0 + b.0, a.1 + b.1)
}

pub fn offset_length(x: (i64, i64)) -> f64 {
    ((x.0 as f64).powi(2) + (x.1 as f64).powi(2)).sqrt()
}

struct RepeatN<T> {
    f: Box<dyn Fn() -> T>,
    counter: usize,
    limit: usize,
}

impl<T> RepeatN<T> {
    pub fn new(n: usize, f: impl Fn() -> T + 'static) -> RepeatN<T> {
        RepeatN {
            f: Box::new(f) as Box<dyn Fn() -> T>,
            counter: 0,
            limit: n,
        }
    }
}

impl<T> Iterator for RepeatN<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.counter == self.limit {
            None
        } else {
            self.counter += 1;
            Some((self.f)())
        }
    }
}

impl<T> Grid<T> {
    pub fn from_factory(height: usize, width: usize, f: impl Fn() -> T + 'static) -> Grid<T> {
        Grid::from(height, width, RepeatN::new(height * width, f))
    }

    pub fn from(height: usize, width: usize, data: impl Iterator<Item = T>) -> Grid<T> {
        Grid {
            stride: width,
            data: data.collect(),
        }
    }

    pub fn from_copy_dimensions<R>(&self, data: impl Iterator<Item = R>) -> Grid<R> {
        Grid {
            stride: self.width(),
            data: data.collect(),
        }
    }

    pub fn width(&self) -> usize {
        self.stride
    }

    pub fn height(&self) -> usize {
        self.data.len() / self.stride
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    pub fn iter_mut_with_indices(&mut self) -> impl Iterator<Item = (&mut T, (usize, usize))> {
        let stride = self.stride;
        self.data
            .iter_mut()
            .enumerate()
            .map(move |(i, x)| (x, (i / stride, i % stride)))
    }

    pub fn iter_with_indices(&self) -> impl Iterator<Item = (&T, (usize, usize))> {
        let stride = self.stride;
        self.data
            .iter()
            .enumerate()
            .map(move |(i, x)| (x, (i / stride, i % stride)))
    }

    pub fn iter_indices(&self) -> impl Iterator<Item = (usize, usize)> {
        let stride = self.stride;
        (0..self.data.len()).map(move |i| (i / stride, i % stride))
    }

    pub fn copy_dimensions<R: Clone>(&self, item: R) -> Grid<R> {
        Grid {
            stride: self.stride,
            data: vec![item; self.data.len()],
        }
    }
    pub fn get_wrapping_index(&self, index: (i64, i64)) -> (usize, usize) {
        (
            index.0.rem_euclid(self.width() as i64) as usize,
            index.1.rem_euclid(self.height() as i64) as usize,
        )
    }

    pub fn get_wrapping(&self, index: (i64, i64)) -> &T {
        &self[(
            index.0.rem_euclid(self.width() as i64) as usize,
            index.1.rem_euclid(self.height() as i64) as usize,
        )]
    }

    pub fn get_wrapping_mut(&mut self, index: (i64, i64)) -> &mut T {
        let width = self.width();
        let height = self.height();

        &mut self[(
            index.0.rem_euclid(width as i64) as usize,
            index.1.rem_euclid(height as i64) as usize,
        )]
    }
}

impl<T: Send> Grid<T> {
    pub fn par_iter_mut_with_indices(
        &mut self,
    ) -> impl ParallelIterator<Item = (&mut T, (usize, usize))> {
        let stride = self.stride;
        self.data
            .par_iter_mut()
            .enumerate()
            .map(move |(i, x)| (x, (i / stride, i % stride)))
    }

    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = &mut T> {
        self.data.par_iter_mut()
    }
}

impl<T: Send + Sync> Grid<T> {
    pub fn par_iter_with_indices(&self) -> impl ParallelIterator<Item = (&T, (usize, usize))> {
        let stride = self.stride;
        self.data
            .par_iter()
            .enumerate()
            .map(move |(i, x)| (x, (i / stride, i % stride)))
    }
}

impl<T: Clone> Grid<T> {
    pub fn new(rows: usize, cols: usize, item: T) -> Grid<T> {
        Grid {
            stride: cols,
            data: vec![item; rows * cols],
        }
    }
}

impl<T> Index<(usize, usize)> for Grid<T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &T {
        if idx.1 >= self.stride || idx.0 * self.stride >= self.data.len() {
            panic!(
                "{:?} is out of range on a grid of size {:?}",
                idx,
                (self.data.len() / self.stride, self.stride)
            )
        }
        self.data.index(idx.0 * self.stride + idx.1)
    }
}

impl<T> IndexMut<(usize, usize)> for Grid<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        if idx.1 >= self.stride || idx.0 * self.stride >= self.data.len() {
            panic!(
                "{:?} is out of range on a grid of size {:?}",
                idx,
                (self.data.len() / self.stride, self.stride)
            )
        }
        self.data.index_mut(idx.0 * self.stride + idx.1)
    }
}

pub fn overlay<A: Clone, B: Clone, R>(a: &Grid<A>, b: &Grid<B>, f: impl Fn(A, B) -> R) -> Grid<R> {
    if a.stride != b.stride || a.data.len() != b.data.len() {
        panic!(
            "Cannot overlay a grid of size {:?} on a grid of size {:?}",
            (a.data.len() / a.stride, a.stride),
            (b.data.len() / b.stride, b.stride)
        );
    }

    Grid {
        data: a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(a, b)| f(a.clone(), b.clone()))
            .collect(),
        stride: a.stride,
    }
}
