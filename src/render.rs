use rayon::prelude::*;
use vecmath::*;

use crate::common::*;
use crate::grid::*;

pub fn compute_normal_at(heights: &Grid<f64>, index: (usize, usize), scale: f64) -> Vector3<f64> {
    // source:
    // https://stackoverflow.com/questions/49640250/calculate-normals-from-heightmap

    let low_x = heights.get_wrapping(shift_index(index, (-1, 0)));
    let high_x = heights.get_wrapping(shift_index(index, (1, 0)));
    let dx = high_x - low_x;

    let low_y = heights.get_wrapping(shift_index(index, (0, -1)));
    let high_y = heights.get_wrapping(shift_index(index, (0, 1)));
    let dy = high_y - low_y;

    vec3_normalized([dx, dy, 2.0 / scale])
}

pub fn compute_relief(heights: &Grid<f64>, to_light: Vector3<f64>, scale: f64) -> Grid<f64> {
    let mut relief = Grid::new(heights.height(), heights.width(), 0.0);
    relief.par_iter_mut_with_indices().for_each(|(val, i)| {
        let normal = compute_normal_at(heights, i, scale);
        let brightness = vec3_dot(normal, to_light).clamp(0.0, 1.0);
        *val = brightness;
    });
    return relief;
}

pub fn gradient(greyscale: &Grid<f64>, colors: &Vec<(f64, (u8, u8, u8))>) -> Grid<(u8, u8, u8)> {
    let mut colored = Grid::new(greyscale.height(), greyscale.width(), (0, 0, 0));
    colored
        .par_iter_mut_with_indices()
        .for_each(|(value, index)| {
            let original = greyscale[index];
            let mut below_mark = 0.0;
            let mut below_color = colors[0].1;
            let mut above_mark = 0.0;
            let mut above_color = colors[0].1;
            for (mark, color) in colors.iter() {
                if *mark <= original {
                    below_mark = *mark;
                    below_color = *color;
                }
                if *mark >= original {
                    above_mark = *mark;
                    above_color = *color;
                    break;
                }
            }
            let percent = (original - below_mark) / (above_mark - below_mark);
            *value = (
                interpolate(below_color.0 as f64, above_color.0 as f64, percent) as u8,
                interpolate(below_color.1 as f64, above_color.1 as f64, percent) as u8,
                interpolate(below_color.2 as f64, above_color.2 as f64, percent) as u8,
            )
        });
    colored
}
