use std::f64::consts::PI;

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum UnitVec {
    Xy(/* normalized x a y coordinates */ f64, f64),
    Theta(/* between 0 and 2pi */ f64),
}

impl UnitVec {
    pub fn from_xy(x: f64, y: f64) -> UnitVec {
        let norm = f64::sqrt(x * x + y * y);
        UnitVec::Xy(x / norm, y / norm)
    }
    pub fn from_angle(theta: f64) -> UnitVec {
        UnitVec::Theta((theta % (2.0 * PI)).abs())
    }
    pub fn get_xy(&self) -> [f64; 2] {
        match self {
            &UnitVec::Xy(x, y) => [x, y],
            &UnitVec::Theta(theta) => [theta.cos(), theta.sin()],
        }
    }
    pub fn get_angle(&self) -> f64 {
        match self {
            &UnitVec::Theta(theta) => theta,
            &UnitVec::Xy(x, y) => {
                let t = f64::atan2(y, x);
                if t < 0.0 {
                    t + 2.0 * PI
                } else {
                    t
                }
            }
        }
    }
}
