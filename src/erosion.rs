use rand::Rng;
use rand::SeedableRng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use std::f64;
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

use crate::grid::*;
use crate::vector::UnitVec;
use crate::ChaCha20Rng;
use num_traits::Float;

use crate::image::*;

pub struct Params {
    pub rain_rate: f64,
    pub evaporation_rate: f64,

    pub min_height_delta: f64,
    pub gravity: f64,

    pub sediment_capacity_constant: f64,
    pub dissolving_rate: f64,
    pub deposition_rate: f64,
}

fn id(x: f64) -> f64 {
    x
}
fn neg(x: f64) -> f64 {
    x
}
fn less(x: f64) -> f64 {
    1.0 - x.abs()
}

fn displace(val: &Grid<f64>, flow: &Grid<UnitVec>) -> Grid<f64> {
    let mut res = val.copy_dimensions(0.0);

    let fns = [id, neg, less];

    for dx in -1i64..=1 {
        let fx = fns[(dx + 1) as usize];
        let wx = flow.from_copy_dimensions_par(flow.par_iter().map(|v| {
            let x = v.get_xy()[0];
            0.0f64.max(fx(x))
        }));
        for dy in -1i64..=1 {
            let fy = fns[(dy + 1) as usize];
            let wy = flow.from_copy_dimensions_par(flow.par_iter().map(|v| {
                let y = v.get_xy()[1];
                0.0f64.max(fy(y))
            }));
            res.par_iter_mut_with_indices().for_each(|(v, idx)| {
                let (x, y) = to_signed_index(idx);
                let idx = (x + dx, y + dy);
                *v += val.get_wrapping(idx) * wx.get_wrapping(idx) * wy.get_wrapping(idx)
            })
        }
    }

    res
}

pub fn erode(
    heightmap: &mut Grid<f64>,
    params: Params,
    iterations: usize,
    seed: usize,
) -> Grid<f64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed as u64);
    let mut water: Grid<f64> = heightmap.copy_dimensions(0.0);
    let mut sediment: Grid<f64> = heightmap.copy_dimensions(0.0);
    let mut velocity: Grid<f64> = heightmap.copy_dimensions(0.0);

    for i in 0..iterations {
        to_gray_image(heightmap)
            .save(format!("debug_heightmap_{:02}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        // Add water to the system
        water
            .par_iter_mut()
            .for_each(|x| *x += 0.5 * params.rain_rate);

        to_gray_image(&water)
            .save(format!("debug_water_{:02}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        dbg!(water.par_iter().sum::<f64>());

        let total_heights = zip(&heightmap, &water, |h, w| h + w);

        // compute the gradient, and the delta height based on that
        let gradient =
            heightmap.from_copy_dimensions(total_heights.convolve().map(|(_, neighbors)| {
                let dx = neighbors.left - neighbors.right;
                let dy = neighbors.down - neighbors.up;
                if dx.abs() < 1e-9 && dy.abs() < 1e-9 {
                    UnitVec::from_angle(rng.gen_range(0.0..(2.0 * PI)))
                } else {
                    UnitVec::from_xy(dx, dy)
                }
            }));

        to_rgb_image(&gradient)
            .save(format!("debug_gradient_{:02}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        let height_delta = zip_with_indices(&total_heights, &gradient, |(h, (x, y)), (g, _)| {
            let [dx, dy] = g.get_xy();
            total_heights.sample((x as f64 - dx, y as f64 - dy)) - h
        });

        // compute the erosion based on velocity, water, and sediment at each location.
        let erosion = heightmap.from_copy_dimensions_par(
            height_delta
                .par_iter()
                .zip(velocity.par_iter())
                .zip(water.par_iter())
                .zip(sediment.par_iter())
                .map(|(((dh, v), w), s)| {
                    let capacity = dh.abs().max(params.min_height_delta)
                        * v.abs()
                        * w.abs()
                        * params.sediment_capacity_constant;
                    if capacity >= *s {
                        if *dh < 0.0 {
                            0.0
                        } else {
                            dh.min(params.dissolving_rate * (capacity - s))
                        }
                    } else {
                        if *dh < 0.0 {
                            // we're in a hole; deposit our entire sediment
                            -s
                                // but don't deposit any more than the
                                // difference in height between us and the next
                                // highest point.
                                .min(-*dh) // h is negative (see if clause)
                                           // we negate h doubly; first to find the maximum
                                           // amount to deposit (the negative of a
                                           // negative), and then to represent the
                                           // deposition as a negative erosion.
                        } else {
                            // otherwise, deposit as much as we can based on the
                            // deposition rate, but once again, we want to
                            // deposit no more than the difference between this
                            // square and the next highest one. We deposit
                            -(params.deposition_rate * (s - capacity)).min(*dh)
                        }
                    }
                }),
        );

        to_gray_image(&erosion)
            .save(format!("debug_erosion_{:02}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        // modify the heightmap and suspended sediment
        sediment
            .par_iter_mut()
            .zip(erosion.par_iter())
            .for_each(|(s, e)| {
                *s += e;
            });

        heightmap
            .par_iter_mut()
            .zip(erosion.par_iter())
            .for_each(|(t, e)| {
                *t -= e;
            });

        water = displace(&water, &gradient);
        sediment = displace(&sediment, &gradient);

        to_gray_image(&sediment)
            .save(format!("debug_sediment_{:02}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        // velocity from the last round is identical to the height difference of
        // this round (not true)
        velocity =
            heightmap.from_copy_dimensions_par(height_delta.par_iter().map(|d| d * params.gravity));

        // evaporate water
        water
            .par_iter_mut()
            .for_each(|w| *w *= 1.0 - params.evaporation_rate);
    }

    water
}

/*
pub fn erode_single_drop(heightmap: &mut Grid<f64>, seed: u64, params: Params) {
    // Create water droplet at random point on map
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let lifetime = max(heightmap.width(), heightmap.height()) * 4;
    let wrap = {
        let hh = heightmap.height();
        let hw = heightmap.width();
        move |pt: (usize, usize)| crate::common::wrap(pt, hw, hh)
    };

    let mut x = rng.gen_range(0..heightmap.width()) as f64 / heightmap.width() as f64;
    let mut y = rng.gen_range(0..heightmap.height()) as f64 / heightmap.height() as f64;

    let mut vx = 0.0;
    let mut vy = 0.0;

    let mut water = 1.0;

    let mut sediment: f64 = 0.0;

    let mut total_eroded = 0.0;
    let mut total_deposited = 0.0;

    for _ in 0..lifetime {
        let (minx, maxx) = wrap(bounds(x * heightmap.width() as f64));
        let (miny, maxy) = wrap(bounds(y * heightmap.height() as f64));
        let distx = (x * heightmap.width() as f64).fract();
        let disty = (x * heightmap.height() as f64).fract();

        // Calculate droplet's height and direction of flow using interpolation

        let miny_height = interpolate(heightmap[(minx, miny)], heightmap[(maxx, miny)], distx);
        let maxy_height = interpolate(heightmap[(minx, maxy)], heightmap[(maxx, maxy)], distx);
        let old_height = interpolate(miny_height, maxy_height, disty);
        let dy = maxy_height - miny_height;

        let minx_height = interpolate(heightmap[(minx, miny)], heightmap[(minx, maxy)], disty);
        let maxx_height = interpolate(heightmap[(maxx, miny)], heightmap[(maxx, maxy)], disty);
        let dx = maxx_height - minx_height;

        // Determine new velocity
        vx = (vx * params.inertia) + (dx * params.gravity);
        vy = (vy * params.inertia) + (dy * params.gravity);

        let norm = (vx * vx + vy * vy).sqrt() * heightmap.width() as f64;
        if norm < f64::EPSILON {
            let theta = rng.gen_range(0.0..(2.0 * PI));
            vx = theta.cos();
            vy = theta.sin();
        }

        let nx = wrap01(x + vx / norm);
        let ny = wrap01(y + vy / norm);

        let new_height = {
            let (minx, maxx) = wrap(bounds(nx * heightmap.width() as f64));
            let (miny, maxy) = wrap(bounds(ny * heightmap.height() as f64));
            let distx = (nx * heightmap.width() as f64).fract();
            let disty = (ny * heightmap.height() as f64).fract();
            let miny_height = interpolate(heightmap[(miny, minx)], heightmap[(miny, maxx)], distx);
            let maxy_height = interpolate(heightmap[(maxy, minx)], heightmap[(maxy, maxx)], distx);
            interpolate(miny_height, maxy_height, disty)
        };
        let dh = new_height - old_height;

        // If the new height is higher than the old height, we need to deposit
        // at the old location.
        if dh > 0.0 {
            let to_deposit = sediment.min(dh);
            deposit(x, y, heightmap, to_deposit);
            sediment -= to_deposit;
            total_deposited += to_deposit;
            vx = 0.0;
            vy = 0.0;
            continue;
        }

        let capacity =
            dh.max(params.min_slope) * water * (vx * vx + vy * vy).sqrt() * params.water_capacity;
        dbg!(capacity);

        if capacity > sediment {
            let to_erode = (params.erosion_speed * (capacity - sediment)).max(dh);
            dbg!(to_erode);
            erode(x, y, heightmap, to_erode);
            sediment += to_erode;
            total_eroded += to_erode;
        } else {
            let to_deposit = sediment - capacity * params.deposition_speed;
            dbg!(to_deposit);
            deposit(x, y, heightmap, to_deposit);
            sediment -= to_deposit;
            total_deposited += to_deposit;
        }

        x = nx;
        y = ny;
        water *= params.evaporation_speed;
    }

    dbg!(sediment, total_deposited, total_eroded);
}

fn erode(x: f64, y: f64, heightmap: &mut Grid<f64>, amount: f64) {
    let (minx, maxx) = wrap(
        bounds(x * heightmap.width() as f64),
        heightmap.width(),
        heightmap.height(),
    );
    let (miny, maxy) = wrap(
        bounds(y * heightmap.height() as f64),
        heightmap.width(),
        heightmap.height(),
    );
    let distx = (x * heightmap.width() as f64).fract();
    let disty = (x * heightmap.height() as f64).fract();

    let xmin_amount = amount * (1.0 - distx);
    let xmax_amount = amount * distx;

    heightmap[(minx, miny)] -= xmin_amount * (1.0 - disty);
    heightmap[(minx, maxy)] -= xmin_amount * disty;
    heightmap[(maxx, miny)] -= xmax_amount * (1.0 - disty);
    heightmap[(maxx, maxy)] -= xmax_amount * disty;
}

fn deposit(x: f64, y: f64, heightmap: &mut Grid<f64>, amount: f64) {
    let (minx, maxx) = wrap(
        bounds(x * heightmap.width() as f64),
        heightmap.width(),
        heightmap.height(),
    );
    let (miny, maxy) = wrap(
        bounds(y * heightmap.height() as f64),
        heightmap.width(),
        heightmap.height(),
    );
    let distx = (x * heightmap.width() as f64).fract();
    let disty = (y * heightmap.height() as f64).fract();

    let xmin_amount = amount * (1.0 - distx);
    let xmax_amount = amount * distx;

    heightmap[(minx, miny)] += xmin_amount * (1.0 - disty);
    heightmap[(minx, maxy)] += xmin_amount * disty;
    heightmap[(maxx, miny)] += xmax_amount * (1.0 - disty);
    heightmap[(maxx, maxy)] += xmax_amount * disty;
}
*/
