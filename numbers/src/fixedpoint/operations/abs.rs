use std::sync::Arc;

use crate::{Abs, BooleanType};

use super::super::FixedPointNumber;
use super::twoscomp::twoscomplement_switch;

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Abs
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn abs(mut self) -> Self::Output {
        let sign = self.msb().clone();
        let b = Arc::make_mut(&mut self.bits);

        twoscomplement_switch(&self.context, &sign, b);
        self
    }
    fn abs_assign(&mut self) {
        let sign = self.msb().clone();
        let b = Arc::make_mut(&mut self.bits);

        twoscomplement_switch(&self.context, &sign, b);
    }
}

impl Abs for f32 {
    type Output = f32;
    fn abs(self) -> Self::Output {
        f32::abs(self)
    }
    fn abs_assign(&mut self) {
        *self = f32::abs(*self);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{Abs, FixedPointNumber, FromWithContext};

    #[test]
    fn f32_abs() {
        let ctx = Arc::new(());

        let calc = |v1: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);

            let enc_res = f1.abs();
            let res: f32 = enc_res.into();

            assert_eq!(res, v1.abs(), "f32_abs {} ", v1);
        };

        calc(0.0);
        calc(-0.0);
        calc(1.0);
        calc(3.5);
        calc(-1.0);
        calc(-2.125);
        calc(4.0);
        calc(-2.5);
        calc(-0.5);
        calc(8.0);
    }
}
