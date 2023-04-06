pub mod sync;
use std::collections::HashSet;
use std::hash::Hash;

pub fn scale(x: f64) -> u8 {
    (x.clamp(0.0, 1.0) * 256.0) as u8
}

pub fn bounds(x: f64) -> (usize, usize) {
    let lower_bound = x.floor() as usize;
    (lower_bound, lower_bound + 1)
}

pub fn interpolate(a: f64, b: f64, x: f64) -> f64 {
    a + x * (b - a)
}

pub fn slope(a: f64, b: f64, x: f64) -> f64 {
    (b - a) * 6.0 * (x - x * x)
}

pub fn wrap(pt: (usize, usize), wrap_x: usize, wrap_y: usize) -> (usize, usize) {
    (
        if pt.0 >= wrap_x { pt.0 - wrap_x } else { pt.0 },
        if pt.1 >= wrap_y { pt.1 - wrap_y } else { pt.1 },
    )
}

pub fn wrap01(x: f64) -> f64 {
    x - x.floor()
}

pub fn choose_and_remove<T: Eq + Hash + Clone>(set: &mut HashSet<T>) -> Option<T> {
    let mut ret = None;

    for x in set.iter() {
        ret = Some(x.clone());
        break;
    }

    if let Some(x) = &ret {
        set.remove(&x);
    }

    ret
}
