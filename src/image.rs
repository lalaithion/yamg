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
        let angle = self.get_angle();

        assert!(
            angle >= 0.0 && angle <= 2.0 * PI,
            "{:?}.get_angle == {}",
            self,
            angle
        );

        let rgb: prisma::Rgb<f64> =
            prisma::Rgb::from_color(&prisma::Hsv::new(Rad(angle), 1.0f64, 1.0));
        rgb.to_tuple().to_rgb()
    }
}

pub enum LumaError {
    TooLow,
    TooHigh,
}

pub trait ToLuma {
    fn to_luma(&self) -> Luma<u8>;
    fn try_to_luma(&self) -> Result<Luma<u8>, LumaError> {
        Ok(self.to_luma())
    }
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

    fn try_to_luma(&self) -> Result<Luma<u8>, LumaError> {
        if *self >= 0.0 && *self <= 1.0 {
            Ok(Luma([scale(*self)]))
        } else if *self > 1.0 {
            Err(LumaError::TooHigh)
        } else {
            Err(LumaError::TooLow)
        }
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

pub fn to_gray_image_handle_outliers(it: &Grid<impl ToLuma>) -> RgbImage {
    let mut img = ImageBuffer::new(it.width() as u32, it.height() as u32);
    for x in 0..it.width() {
        for y in 0..it.height() {
            match it[(x, y)].try_to_luma() {
                Ok(luma) => img.put_pixel(
                    x as u32,
                    y as u32,
                    (luma.0[0], luma.0[0], luma.0[0]).to_rgb(),
                ),
                Err(LumaError::TooLow) => img.put_pixel(x as u32, y as u32, (0, 0, 255).to_rgb()),
                Err(LumaError::TooHigh) => img.put_pixel(x as u32, y as u32, (255, 0, 0).to_rgb()),
            }
        }
    }
    img
}
