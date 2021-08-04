pub mod sync;
use std::collections::HashSet;
use std::hash::Hash;

pub fn scale(x: f64) -> u8 {
    (x.clamp(0.0, 1.0) * 256.0) as u8
}

pub fn bounds(x: f64) -> (u64, u64) {
    let lower_bound = x.floor() as u64;
    (lower_bound, lower_bound + 1)
}

pub fn interpolate(a: f64, b: f64, x: f64) -> f64 {
    //(b - a) * x + a
    (b - a) * (3.0 - x * 2.0) * x * x + a
}

pub fn wrap(pt: (usize, usize), wrap_x: usize, wrap_y: usize) -> (usize, usize) {
    (
        if pt.0 == wrap_x { 0 } else { pt.0 },
        if pt.1 == wrap_y { 0 } else { pt.1 },
    )
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