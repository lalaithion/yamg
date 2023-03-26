use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::f64::consts::PI;
use vecmath::*;

use crate::common::*;
use crate::grid::*;
use crate::vector::*;

#[derive(Debug)]
pub struct Perlin {
    octaves: u8,
    scale: u8,
    vectors: Vec<Grid<UnitVec>>,
}

impl Perlin {
    pub fn new(octaves: u8, scale: u8, seed: u64) -> Perlin {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        dbg!(rng.gen::<u8>());
        let mut vectors = Vec::with_capacity(octaves as usize);
        for i in 0..octaves {
            let grid_size = scale as usize * 2usize.pow(i as u32);
            let mut grid = Grid::new(grid_size, grid_size, UnitVec::Theta(0.0));
            for x in 0..grid_size {
                for y in 0..grid_size {
                    grid[(x, y)] = UnitVec::Theta(rng.gen_range(0.0..2.0 * PI));
                }
            }
            vectors.push(grid);
        }
        Perlin {
            octaves,
            scale,
            vectors,
        }
    }

    pub fn sample(&self, position: Vector2<f64>) -> f64 {
        let mut accumulator = 0.0;
        for (octave, grid) in self.vectors.iter().enumerate() {
            let grid_size = self.scale as usize * 2usize.pow(octave as u32);
            let scaled_position = vec2_scale(position, grid_size as f64);

            let (low_x, high_x) = bounds(scaled_position[0]);
            let (low_y, high_y) = bounds(scaled_position[1]);
            let corners = vec![
                [low_x, low_y],
                [low_x, high_y],
                [high_x, low_y],
                [high_x, high_y],
            ]
            .into_iter();

            let vectors = corners
                .clone()
                .map(|[x, y]| grid[wrap((x, y), grid_size, grid_size)]);

            let distances = corners.map(|v| vec2_sub([v[0] as f64, v[1] as f64], scaled_position));

            let dots: Vec<f64> = vectors
                .zip(distances)
                .map(|(c, d)| vec2_dot(c.get_xy(), d))
                .collect();

            let a0 = interpolate(dots[0], dots[1], scaled_position[1].fract());
            let a1 = interpolate(dots[2], dots[3], scaled_position[1].fract());
            let res = interpolate(a0, a1, scaled_position[0].fract());
            accumulator += res * 2.0f64.powf(-(octave as f64));
        }
        (accumulator + 1.0) / 2.0
    }

    pub fn render_to(&self, size: usize) -> Grid<f64> {
        let mut grid = Grid::new(size, size, 0.0);
        grid.par_iter_mut_with_indices().for_each(|(val, (x, y))| {
            let position = [x as f64 / size as f64, y as f64 / size as f64];
            *val = self.sample(position);
        });
        return grid;
    }
}
