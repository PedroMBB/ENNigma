use crate::{BooleanType, FixedPointNumber, FromWithContext, IsGreater, Max, Min};

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Max
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn max(&self, rhs: &Self) -> Self {
        let false_val: T = FromWithContext::from_ctx(false, &self.context);
        let dif = T::xor(self.msb(), rhs.msb());

        let is_greater = self
            .bits
            .iter()
            .zip(rhs.bits.iter())
            .fold(false_val, |c, (v1, v2)| {
                let xnor_res = T::xnor_ref(v1, v2);
                xnor_res.mux(&c, v1)
            })
            ^ dif;

        Self::from_bits(
            self.bits
                .iter()
                .zip(rhs.bits.iter())
                .map(|(v1, v2)| is_greater.mux(v1, v2))
                .collect(),
            &self.context,
        )
    }
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Min
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn min(&self, rhs: &Self) -> Self {
        let false_val: T = FromWithContext::from_ctx(false, &self.context);
        let dif = T::xor(self.msb(), rhs.msb());

        let is_greater = self
            .bits
            .iter()
            .zip(rhs.bits.iter())
            .fold(false_val, |c, (v1, v2)| {
                let xnor_res = T::xnor_ref(v1, v2);
                xnor_res.mux(&c, v1)
            })
            ^ dif;

        Self::from_bits(
            self.bits
                .iter()
                .zip(rhs.bits.iter())
                .map(|(v1, v2)| is_greater.mux(v2, v1))
                .collect(),
            &self.context,
        )
    }
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> IsGreater
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn is_greater(&self, rhs: &Self) -> Self {
        let false_val: T = FromWithContext::from_ctx(false, &self.context);
        let dif = T::xor(self.msb(), rhs.msb());
        let is_greater =
            self.bits
                .iter()
                .zip(rhs.bits.iter())
                .fold(false_val.clone(), |c, (v1, v2)| {
                    let xnor_res = T::xnor_ref(v1, v2);
                    xnor_res.mux(&c, v1)
                })
                ^ dif;

        let mut bits: Vec<T> = (0..SIZE).map(|_| false_val.clone()).collect();
        bits[PRECISION] = is_greater;

        Self::from_bits(bits, &self.context)
    }
}

impl IsGreater for f32 {
    fn is_greater(&self, rhs: &Self) -> Self {
        match self > rhs {
            true => 1.0,
            false => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{FixedPointNumber, FromWithContext, IsGreater, Max, Min};

    #[test]
    fn f32_max() {
        let ctx = Arc::new(());

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let v1_r: f32 = f1.clone().into();
            let v2_r: f32 = f2.clone().into();

            let enc_res = f1.max(&f2);
            let res: f32 = enc_res.into();

            assert_eq!(res, v1.max(v2), "f32_max {} ~ {} ", v1_r, v2_r);
        };

        calc(3.5, 1.0);
        calc(1.0, 1.0);
        calc(0.0, 0.0);
        calc(-2.5, -3.0);
        calc(-2.5, 2.0);
        calc(-0.0, 1.0);
        calc(0.0, -1.0);
        calc(-1.0, 1.0);
        calc(-2.0, 1.5);
        calc(4.0, -3.0);
        calc(-0.5, -1.0);
        calc(8.0, 1.0);
        calc(15.0, -15.0);
    }

    #[test]
    fn f32_min() {
        let ctx = Arc::new(());

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res = f1.min(&f2);
            let res: f32 = enc_res.into();

            assert_eq!(res, v1.min(v2), "f32_min {} ~ {} ", v1, v2);
        };

        calc(3.5, 1.0);
        calc(1.0, 1.0);
        calc(0.0, 0.0);
        calc(-2.5, -3.0);
        calc(-2.5, 2.0);
        calc(-0.0, 1.0);
        calc(0.0, -1.0);
        calc(-1.0, 1.0);
        calc(-2.0, 1.5);
        calc(4.0, -3.0);
        calc(-0.5, -1.0);
        calc(8.0, 1.0);
    }

    #[test]
    fn f32_isgreater() {
        let ctx = Arc::new(());

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res = f1.is_greater(&f2);
            let res: f32 = enc_res.into();

            assert_eq!(
                res,
                match v1 > v2 {
                    true => 1.0,
                    false => 0.0,
                },
                "f32_isgreater {} ~ {} ",
                v1,
                v2
            );
        };

        calc(3.5, 1.0);
        calc(1.0, 1.0);
        calc(0.0, 0.0);
        calc(-2.5, -3.0);
        calc(-2.5, 2.0);
        calc(-0.0, 1.0);
        calc(0.0, -1.0);
        calc(-1.0, 1.0);
        calc(-2.0, 1.5);
        calc(4.0, -3.0);
        calc(-0.5, -1.0);
        calc(8.0, 1.0);
    }
}
