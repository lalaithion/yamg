use std::f64;
use std::f64::consts::PI;

use indicatif::ProgressBar;
use rand::Rng;
use rand::SeedableRng;
use rayon::iter::ParallelIterator;

use crate::common::interpolate;
use crate::grid::*;
use crate::vector::UnitVec;
use crate::ChaCha20Rng;

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
    -x
}
fn less(x: f64) -> f64 {
    1.0 - x.abs()
}

fn displace(val: &Grid<f64>, flow: &Grid<UnitVec>) -> Grid<f64> {
    let mut res = val.copy_dimensions(0.0);

    let fns = [neg, less, id];

    let wxs: Vec<Grid<f64>> = (-1i64..=1)
        .map(|dx| {
            let fx = fns[(dx + 1) as usize];
            flow.from_copy_dimensions(flow.iter().map(|v| {
                let x = v.get_xy()[0];
                0.0f64.max(fx(x))
            }))
        })
        .collect();

    let wys: Vec<Grid<f64>> = (-1i64..=1)
        .map(|dy| {
            let fy = fns[(dy + 1) as usize];
            flow.from_copy_dimensions_par(flow.par_iter().map(|v| {
                let y = v.get_xy()[1];
                0.0f64.max(fy(y))
            }))
        })
        .collect();

    res.par_iter_mut_with_indices().for_each(|(v, idx)| {
        for dx in -1i64..=1 {
            let wx = &wxs[(dx + 1) as usize];
            for dy in -1i64..=1 {
                let wy = &wys[(dy + 1) as usize];
                let (x, y) = to_signed_index(idx);
                let idx = val.get_wrapping_index((x + dx, y + dy));
                *v += val[idx] * wx[idx] * wy[idx]
            }
        }
    });

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

    let bar = ProgressBar::new(iterations as u64);

    for i in 0..iterations {
        bar.inc(1);

        // Add water to the system
        water.iter_mut().for_each(|x| *x += 0.5 * params.rain_rate);

        // compute the gradient, and the delta height based on that
        let gradient =
            heightmap.from_copy_dimensions(heightmap.convolve().map(|(_, neighbors)| {
                let dx = neighbors.left - neighbors.right;
                let dy = neighbors.up - neighbors.down;
                if dx.abs() < 1e-9 && dy.abs() < 1e-9 {
                    UnitVec::from_angle(rng.gen_range(0.0..(2.0 * PI)))
                } else {
                    UnitVec::from_xy(-dx, -dy)
                }
            }));

        let height_delta = zip_with_indices(&heightmap, &gradient, |(h, (x, y)), (g, _)| {
            let [dx, dy] = g.get_xy();
            (h - heightmap.sample((x as f64 - dx, y as f64 - dy))).min(1.0)
        });

        // compute the erosion based on velocity, water, and sediment at each location.
        let erosion = heightmap.from_copy_dimensions(
            height_delta
                .iter()
                .zip(velocity.iter())
                .zip(water.iter())
                .zip(sediment.iter())
                .map(|(((dh, v), w), s)| {
                    let capacity = dh.abs().max(params.min_height_delta)
                        * v
                        * w
                        * params.sediment_capacity_constant;
                    if *dh < 0.0 {
                        (-s).min(*dh)
                    } else if *s > capacity {
                        (params.deposition_rate * (*s - capacity)).min(*dh)
                    } else {
                        params.dissolving_rate * (*s - capacity).min(*dh)
                    }
                }),
        );

        let greyscale_erosion =
            erosion.from_copy_dimensions(erosion.iter().map(|x| (x + 1.0) * 0.5));
        to_gray_image_handle_outliers(&greyscale_erosion)
            .save(format!("debug_erosion_{:04}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        // modify the heightmap and suspended sediment
        sediment.iter_mut().zip(erosion.iter()).for_each(|(s, e)| {
            *s += e;
        });

        heightmap.iter_mut().zip(erosion.iter()).for_each(|(t, e)| {
            *t -= e;
        });

        *heightmap =
            heightmap.from_copy_dimensions(heightmap.convolve_indices().map(|(x, idx, n)| {
                interpolate(
                    *x,
                    (x + (n.up + n.down + n.left + n.right) / 4.0) / 2.0,
                    height_delta[idx].abs(),
                )
            }));

        to_gray_image_handle_outliers(&heightmap)
            .save(format!("debug_heightmap_{:04}.png", i))
            .unwrap_or_else(|e| eprintln!("{}", e));

        water = displace(&water, &gradient);
        sediment = displace(&sediment, &gradient);

        // velocity for the next round is identical to the height difference of
        // this round (not true)
        velocity = height_delta.clone();

        // evaporate water
        water
            .iter_mut()
            .for_each(|w| *w *= 1.0 - params.evaporation_rate);
    }
    bar.finish();

    water
}
