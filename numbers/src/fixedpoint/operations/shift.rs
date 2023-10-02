use std::ops::{Shl, Shr};

use crate::{BooleanType, FixedPointNumber};

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Shl<usize>
    for &FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn shl(self, rhs: usize) -> Self::Output {
        let false_v = T::from_ctx(false, &self.context);

        let mut res: Vec<_> = self.bits.iter().map(|v| v).collect();
        let mut other_vec: Vec<_> = (0..rhs).map(|_| &false_v).collect();
        other_vec.append(&mut res);

        Self::Output::from_iter(other_vec.into_iter().cloned(), &self.context)
    }
}
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Shr<usize>
    for &FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn shr(self, rhs: usize) -> Self::Output {
        let sign = self.msb();
        let mut res: Vec<_> = self
            .bits
            .iter()
            .skip(rhs)
            .take(SIZE - rhs - 1)
            .map(|v| v)
            .collect();
        let mut other_vec: Vec<_> = (0..rhs).map(|_| sign).collect();
        res.append(&mut other_vec);
        res.push(sign);

        Self::Output::from_iter(res.into_iter().cloned(), &self.context)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{FixedPointNumber, FromWithContext};

    #[test]
    fn f32_shl() {
        let ctx = Arc::new(());

        let calc = |v1: f32, n: usize| {
            let f1: FixedPointNumber<8, 3, bool, _> = FixedPointNumber::from_ctx(v1, &ctx);

            let enc_res = &f1 << n;
            let res: f32 = enc_res.into();

            assert_eq!(
                res,
                v1 * 2.0_f32.powi(n.try_into().expect("Should convert")),
                "f32_shl {} << {} ",
                v1,
                n
            );
        };

        calc(0.0, 1);
        calc(-0.0, 2);
        calc(-0.0, 3);
        calc(1.0, 0);
        calc(1.0, 2);
        calc(3.5, 1);
        calc(-1.0, 1);
    }

    #[test]
    fn f32_shr() {
        let ctx = Arc::new(());

        let calc = |v1: f32, n: usize| {
            let f1: FixedPointNumber<8, 3, bool, _> = FixedPointNumber::from_ctx(v1, &ctx);

            let enc_res = &f1 >> n;
            let res: f32 = enc_res.into();

            assert_eq!(
                res,
                v1 / 2.0_f32.powi(n.try_into().unwrap()),
                "f32_shr {} >> {} ",
                v1,
                n
            );
        };

        calc(0.0, 1);
        calc(-0.0, 2);
        calc(-0.0, 3);
        calc(1.0, 0);
        calc(1.0, 2);
        calc(3.5, 1);
        calc(-1.0, 1);
    }
}
