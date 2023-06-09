#![allow(dead_code)]

mod common;
mod erosion;
mod grid;
mod image;
mod perlin;
mod render;
mod vector;

use ::image::Rgb;
use itertools::Itertools;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use std::cmp::Ordering::*;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::f64::consts::E;
use std::time::Instant;
use vecmath::*;

use crate::image::*;
use common::*;
use grid::*;
use perlin::*;
use render::*;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
struct FlowState {
    dir: (i64, i64),
    flow: f64,
    is_lake: bool,
}

fn compute_water_flows(height_map: &Grid<Lake>) -> Grid<FlowState> {
    let mut indices: Vec<(usize, usize)> = height_map.iter_indices().collect();
    indices.sort_unstable_by(|a, b| {
        height_map[*b]
            .height
            .partial_cmp(&height_map[*a].height)
            .unwrap_or(Equal)
    });

    let mut flow = height_map.copy_dimensions(FlowState {
        dir: (0, 0),
        flow: 0.0,
        is_lake: false,
    });

    for index in indices {
        let mut outgoing_flow = 0.0;
        let current_height = height_map[index].height;
        let mut min_height = current_height;
        let mut min_dir = (0, 0);

        for offset in NEIGHBOR_OFFSETS.iter() {
            let target_height = height_map.get_wrapping(shift_index(index, *offset)).height;
            let target_flow = flow.get_wrapping(shift_index(index, *offset));
            if target_height < min_height {
                min_height = target_height;
                min_dir = *offset;
            } else if target_height > current_height
                && add_offsets(target_flow.dir, *offset) == (0, 0)
            {
                outgoing_flow += target_flow.flow;
            }
        }

        outgoing_flow += 0.1;

        flow[index] = FlowState {
            dir: min_dir,
            flow: outgoing_flow,
            is_lake: height_map[index].is_lake,
        };
    }

    flow
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
struct Lake {
    height: f64,
    is_lake: bool,
    lake_id: u16,
    lake_out: (usize, usize),
}

impl ToRgb for Lake {
    fn to_rgb(&self) -> ::image::Rgb<u8> {
        let height = (255.0 * self.height) as u8;
        if self.is_lake {
            Rgb([0, height, 255])
        } else {
            Rgb([height, height, height])
        }
    }
}

fn compute_lakes(height_map: &Grid<f64>) -> (Grid<Lake>, u16) {
    // We start by making a grid the size of the height map, with each element
    // not being a lake, and their height at the max height.
    let mut lakes = height_map.copy_dimensions(Lake {
        height: 1.0,
        is_lake: false,
        lake_id: 0,
        lake_out: (0, 0),
    });

    // Then, we'll use the oceans (0.4) as our initial heights.
    for (v, i) in height_map.iter_with_indices() {
        if *v < 0.4 {
            lakes[i] = Lake {
                height: *v,
                is_lake: false,
                lake_id: 0,
                lake_out: i,
            }
        }
    }

    // Do the following loop until it's stable.

    // TODO(izaakweiss): There should be a way to convince Rust that we're
    // only writing to one index per for_each, and therefore don't need to have
    // copy the entire grid on each iteration, which would shave down processing
    // time. This loop is currently the most expensive part of the entire program.
    let mut changed = true;
    while changed {
        let mut new_lakes = lakes.clone();
        new_lakes.iter_mut_with_indices().for_each(|(v, i)| {
            let mut min_height = lakes[i].height;
            for offset in NEIGHBOR_OFFSETS {
                let lake = lakes.get_wrapping(shift_index(i, offset));
                if lake.height < min_height {
                    min_height = lake.height;
                }
            }

            if min_height == lakes[i].height {
                // if it doesn't have any lower neighbors, do nothing.
            } else if height_map[i] >= min_height && lakes[i].height != height_map[i] {
                // if it does have a lower neighbor, and our original height was
                // higher than that neighbor, set the lake height equal to the
                // original height (and we're not in a lake, so is_lake stays
                // false). We also check to see if we have already set this
                // value so we can set changed.
                v.height = height_map[i];
                v.is_lake = false;
                //changed.store(true, Ordering::Relaxed);
            } else if height_map[i] < min_height
                && lakes[i].height != height_map[i]
                && lakes[i].height != min_height + 0.0001
            {
                // if it does have a lower neighbor, but the original height
                // would be lower than the lowest neighbor, we're in a
                // depression, so we're in a lake! We set the lake height to a
                // little higher than the lowest neighbor, and mark that we're
                // in a lake. We also check to see if we have already set this
                // value so we can set changed.
                v.height = min_height + 0.0001;
                v.is_lake = true;
                //changed.store(true, Ordering::Relaxed);
            }
        });
        changed = lakes != new_lakes;
        lakes = new_lakes;
    }

    let lake_count = label_lakes(&mut lakes);

    (lakes, lake_count)
}

fn label_lakes(lakes: &mut Grid<Lake>) -> u16 {
    let mut queue = VecDeque::new();
    let mut unprocessed: HashSet<(usize, usize)> = lakes
        .iter_with_indices()
        .filter_map(|(v, i)| if v.is_lake { Some(i) } else { None })
        .collect();
    let mut outputs = vec![(0, 0)];

    let mut lake_id = 0;

    while let Some(start_point) = choose_and_remove(&mut unprocessed).clone() {
        queue.push_front(start_point);
        lakes[start_point].lake_id = lake_id;
        lake_id += 1;
        outputs.push((0, 0));
        let mut out_candidate_height = lakes[start_point].height;

        while let Some(index) = queue.pop_front() {
            for offset in NEIGHBOR_OFFSETS {
                let neighbor_index = lakes.get_wrapping_index(shift_index(index, offset));
                if unprocessed.contains(&neighbor_index) {
                    unprocessed.remove(&neighbor_index);
                    if lakes[neighbor_index].is_lake {
                        lakes[neighbor_index].lake_id = lakes[index].lake_id;
                        queue.push_back(neighbor_index);
                    }
                } else if lakes[neighbor_index].height < out_candidate_height {
                    out_candidate_height = lakes[neighbor_index].height;
                    outputs[lake_id as usize] = neighbor_index;
                }
            }
        }
    }

    lakes
        .iter_mut()
        .for_each(|lake| lake.lake_out = outputs[lake.lake_id as usize]);

    lake_id
}

fn draw_lake_labels(labels: &Grid<Lake>, seed: u64) -> Grid<(u8, u8, u8)> {
    let mut res = labels.copy_dimensions((0, 0, 0));
    res.iter_mut_with_indices().for_each(|(v, i)| {
        if labels[i].lake_id == 0 {
            *v = (0, 0, 0)
        } else {
            let mut rng = ChaCha20Rng::seed_from_u64(seed + labels[i].lake_id as u64);
            *v = (
                rng.gen_range(0..255),
                rng.gen_range(0..255),
                rng.gen_range(0..255),
            );
        }
    });

    labels
        .iter()
        .map(|lake| (lake.lake_id, lake.lake_out))
        .unique()
        .for_each(|(id, out)| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed + id as u64);
            res[out] = (
                255 - rng.gen_range(0..255),
                255 - rng.gen_range(0..255),
                255 - rng.gen_range(0..255),
            )
        });

    res
}

fn erode(height_map: &Grid<f64>, flows: &Grid<FlowState>, scale: f64) -> Grid<f64> {
    let mut eroded = height_map.clone();
    eroded.iter_mut_with_indices().for_each(|(v, idx)| {
        let flow = flows[idx].clone();
        let height_diff = NEIGHBOR_OFFSETS
            .iter()
            .map(|offset| height_map.get_wrapping(shift_index(idx, *offset)))
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(Equal))
            .unwrap()
            - height_map[idx];
        if height_diff > flow.flow && flow.dir != (0, 0) {
            let slope = scale.sqrt()
                * (height_map[idx] - height_map.get_wrapping(shift_index(idx, flow.dir)))
                / offset_length(flow.dir);
            *v -= 0.5 * flow.flow.powf(0.8) * slope.powf(2.0);
        } else if flow.dir == (0, 0) || flow.is_lake {
            *v += height_diff / 2.0;
        } else {
            *v += height_diff / (1.0 + E.powf(flow.flow - 6.0));
        }
    });
    eroded
}

fn erode_loop(height_map: &Grid<f64>, iterations: usize) -> Grid<f64> {
    let mut eroded = height_map.clone();
    let mut remaining_water = crate::erosion::erode(
        &mut eroded,
        crate::erosion::Params {
            rain_rate: 0.0008,
            evaporation_rate: 0.0005,

            min_height_delta: 0.05,
            gravity: 30.0,

            sediment_capacity_constant: 1.0,
            dissolving_rate: 0.25,
            deposition_rate: 0.001,
        },
        iterations,
        0,
    );

    let max_water = *remaining_water
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .unwrap_or(&1.0);

    remaining_water.iter_mut().for_each(|x| *x /= max_water);

    to_gray_image(&remaining_water)
        .save("water.png")
        .expect("water.png failed to save");

    eroded
}

fn overlay_rivers(
    background: &Grid<(u8, u8, u8)>,
    height_map: &Grid<f64>,
    sea_level: f64,
    flows: &Grid<FlowState>,
) -> Grid<(u8, u8, u8)> {
    let max_flow = flows
        .iter_with_indices()
        .filter(|(_, i)| height_map[*i] > sea_level)
        .map(|(f, _)| f.flow)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
        .unwrap_or(1.0);

    flows.from_copy_dimensions(
        flows
            .iter()
            .zip(background.iter())
            .zip(height_map.iter())
            .map(|((flow, original), height)| {
                let percent = if flow.is_lake {
                    0.8
                } else {
                    (flow.flow / max_flow).powf(0.2)
                };
                if *height < sea_level {
                    *original
                } else {
                    (
                        ((original.0 as f64 * (1.0 - percent)) + 31.0 * percent) as u8,
                        ((original.1 as f64 * (1.0 - percent)) + 192.0 * percent) as u8,
                        ((original.2 as f64 * (1.0 - percent)) + 207.0 * percent) as u8,
                    )
                }
            }),
    )
}

macro_rules! time {
    ($s:literal, $e:expr) => {{
        let start = Instant::now();
        let x = $e;
        let duration = start.elapsed();
        println!("{} duration: {:?}", $s, duration);
        x
    }};
}

fn main() {
    let p = Perlin::new(7, 4, 0);
    let height_map = time!("perlin", p.render_to(500));
    let terrain_gradient = vec![
        (0.2, (9, 9, 121)),
        (0.4, (31, 192, 207)),
        (0.40001, (231, 222, 31)),
        (0.41, (167, 218, 48)),
        (0.8, (31, 171, 69)),
        (0.9, (145, 125, 46)),
        (1.0, (255, 255, 255)),
    ];

    let eroded = time!("erode", erode_loop(&height_map, 500));

    let lakes = &compute_lakes(&eroded).0;
    let flows = compute_water_flows(lakes);
    let relief = compute_relief(&eroded, vec3_normalized([2.0, 2.0, 4.0]), 25.0);
    let color = overlay(
        &gradient(&eroded, &terrain_gradient),
        &relief,
        |(r, g, b), x| {
            (
                (r as f64 * x) as u8,
                (g as f64 * x) as u8,
                (b as f64 * x) as u8,
            )
        },
    );
    let map = overlay_rivers(&color, &eroded, 0.4, &flows);

    to_rgb_image(&map)
        .save("final.png")
        .expect("final.png failed to save");
    to_rgb_image(&draw_lake_labels(lakes, 1))
        .save("lakes.png")
        .expect("lakes.png failed to save");
}

#[cfg(test)]
mod test {
    use super::*;
    use ::image::io::Reader;
    use std::path::PathBuf;

    #[test]
    fn compare_goldens() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let p = Perlin::new(7, 2, 4);
        let height_map = p.render_to(500);
        let relief_map = compute_relief(&height_map, vec3_normalized([2.0, 2.0, 4.0]), 25.0);
        let height_image = to_gray_image(&height_map);
        let relief_image = to_gray_image(&relief_map);

        let mut golden_height_path = root.clone();
        golden_height_path.push("goldens/height_map.png");
        let golden_height_path = golden_height_path.as_path();

        let golden_height_image = Reader::open(golden_height_path)
            .expect("goldens/height_map.png failed to open")
            .decode()
            .expect("goldens/height_map.png had an incorrect encoding")
            .to_luma8();

        let mut golden_relief_path = root.clone();
        golden_relief_path.push("goldens/relief_map.png");
        let golden_relief_path = golden_relief_path.as_path();
        let golden_relief_image = Reader::open(golden_relief_path)
            .expect("goldens/relief_map failed to open")
            .decode()
            .expect("goldens/relief_map.png had an incorrect encoding")
            .to_luma8();

        assert_eq!(height_image, golden_height_image);
        assert_eq!(relief_image, golden_relief_image);
    }
}
