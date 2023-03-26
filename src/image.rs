#![allow(dead_code)]

use std::f64::consts::PI;

use crate::common::*;
use crate::grid::*;
use crate::vector::UnitVec;
use angular_units::Rad;
use image::*;
use prisma;
use prisma::Color;
use prisma::FromColor;

pub trait ToRgb {
    fn to_rgb(&self) -> Rgb<u8>;
}

impl ToRgb for Rgb<u8> {
    fn to_rgb(&self) -> Rgb<u8> {
        self.clone()
    }
}

impl ToRgb for (u8, u8, u8) {
    fn to_rgb(&self) -> Rgb<u8> {
        Rgb([self.0, self.1, self.2])
    }
}

impl ToRgb for (f64, f64, f64) {
    fn to_rgb(&self) -> Rgb<u8> {
        Rgb([scale(self.0), scale(self.1), scale(self.2)])
    }
}

impl ToRgb for UnitVec {
    fn to_rgb(&self) -> Rgb<u8> {
        let rgb: prisma::Rgb<f64> =
            prisma::Rgb::from_color(&prisma::Hsv::new(Rad(self.get_angle()), 1.0f64, 1.0));
        rgb.to_tuple().to_rgb()
    }
}

pub trait ToLuma {
    fn to_luma(&self) -> Luma<u8>;
}

impl ToLuma for Luma<u8> {
    fn to_luma(&self) -> Luma<u8> {
        self.clone()
    }
}

impl ToLuma for u8 {
    fn to_luma(&self) -> Luma<u8> {
        Luma([*self])
    }
}

impl ToLuma for f64 {
    fn to_luma(&self) -> Luma<u8> {
        Luma([scale(*self)])
    }
}

pub fn to_rgb_image(it: &Grid<impl ToRgb>) -> RgbImage {
    let mut img = ImageBuffer::new(it.width() as u32, it.height() as u32);
    for x in 0..it.width() {
        for y in 0..it.height() {
            img.put_pixel(x as u32, y as u32, it[(x, y)].to_rgb());
        }
    }
    img
}

pub fn to_gray_image(it: &Grid<impl ToLuma>) -> GrayImage {
    let mut img = ImageBuffer::new(it.width() as u32, it.height() as u32);
    for x in 0..it.width() {
        for y in 0..it.height() {
            img.put_pixel(x as u32, y as u32, it[(x, y)].to_luma());
        }
    }
    img
}
