use std::{collections::VecDeque, ops::Div, sync::Arc};

use crate::{
    Abs, BooleanType, DefaultWithContext, DivRemainder, FixedPointNumber, FromWithContext,
};

use super::{addition::add_vectors_assign_carry, twoscomp::twoscomplement_vectors};

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    Div<&FixedPointNumber<SIZE, PRECISION, T, C>> for FixedPointNumber<SIZE, PRECISION, T, C>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, C>;

    fn div(self, rhs: &FixedPointNumber<SIZE, PRECISION, T, C>) -> Self::Output {
        let (q, _) = DivRemainder::div_remainder(self, rhs);
        q
    }
}
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    Div<&FixedPointNumber<SIZE, PRECISION, T, C>> for &FixedPointNumber<SIZE, PRECISION, T, C>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, C>;

    fn div(self, rhs: &FixedPointNumber<SIZE, PRECISION, T, C>) -> Self::Output {
        self.clone() / rhs
    }
}
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    DivRemainder<&FixedPointNumber<SIZE, PRECISION, T, C>>
    for FixedPointNumber<SIZE, PRECISION, T, C>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, C>;

    fn div_remainder(
        self,
        rhs: &FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (Self::Output, Self::Output) {
        let ctx = Arc::clone(&self.context);
        let false_val: T = FromWithContext::from_ctx(false, &ctx);
        let sign = T::xor(self.msb(), rhs.msb());

        let m = self.abs();
        let d = rhs.clone().abs();

        let b: Vec<_> = d.bits.iter().take(SIZE - 1).cloned().collect();
        let mut b_complements = b.clone();
        twoscomplement_vectors(&ctx, &mut b_complements);

        let empty: Self::Output = DefaultWithContext::default_ctx(&ctx);

        let mut aq = VecDeque::with_capacity(SIZE * 2 - 1);
        aq.push_front(false_val.clone());
        aq.extend(empty.bits.iter().take(SIZE - 1).rev().cloned());
        aq.extend(m.bits.iter().take(SIZE - 1).rev().cloned());

        for _ in 0..(SIZE + PRECISION - 1) {
            aq.pop_front();

            let a: Vec<T> = aq
                .iter()
                .skip(1)
                .take(SIZE - 1)
                .map(|v| v.clone())
                .rev()
                .collect();

            let mut comp_a = a.clone();
            let not_e = add_vectors_assign_carry(&ctx, &mut comp_a, b_complements.iter().collect());

            let _r: Vec<_> = aq
                .iter_mut()
                .skip(1)
                .take(SIZE - 1)
                .zip(a.iter().rev())
                .zip(comp_a.iter().rev())
                .map(|((dest, rest_a), a)| {
                    *dest = not_e.mux(a, rest_a);
                })
                .collect::<Vec<_>>();

            aq.push_back(not_e);
        }

        let mut m_vec: Vec<T> = aq.iter().skip(SIZE).take(SIZE - 1).rev().cloned().collect();
        m_vec.push(false_val.clone());
        let m: Self::Output = Self::Output::from_bits(m_vec, &ctx);

        let mut r_vec: Vec<T> = aq.iter().skip(1).take(SIZE - 1).rev().cloned().collect();
        r_vec.push(false_val);
        let remainder: Self::Output = Self::Output::from_bits(r_vec, &ctx);

        let m = &m;

        let not_m = !m.clone();

        let new_bits: Vec<_> = m
            .bits
            .iter()
            .zip(not_m.bits.iter())
            .map(|(a, b)| sign.mux(b, a))
            .collect();

        (Self::Output::from_bits(new_bits, &ctx), remainder)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{FixedPointNumber, FromWithContext};

    #[test]
    fn f32_div() {
        let ctx = Arc::new(());

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res = f1 / &f2;
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 / v2, "f32_div {} / {} ", v1, v2);
        };

        calc(0.0, 1.0);
        calc(2.0, 2.0);
        calc(3.0, 2.0);
        calc(-1.0, -1.0);
        calc(1.0, -1.0);
        calc(2.0, 0.5);
        calc(-2.0, 0.5);
        calc(4.0, -2.0);
        calc(-3.0, 2.0);
        calc(3.0, -2.0);
        calc(-1.0, 2.0);
        calc(6.0, 2.0);
        calc(3.0, 0.5);
        calc(1.5, -0.5);
    }
}
