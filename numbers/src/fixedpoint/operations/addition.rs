use std::{
    ops::{Add, AddAssign, Sub, SubAssign},
    sync::Arc,
};

use crate::{BooleanType, FixedPointNumber};

pub(crate) fn add_vectors_assign_carry<T: BooleanType<C>, C>(
    ctx: &Arc<C>,
    lst1: &mut [T],
    lst2: Vec<&T>,
) -> T {
    lst1.iter_mut()
        .zip(lst2.into_iter())
        .fold::<T, _>(T::from_ctx(false, ctx), |c, (v1, v2)| {
            let mut xor_val = T::xor(v1, v2);
            let mut ci = T::and(&c, &xor_val);

            T::xor_assign(&mut xor_val, &c);
            T::or_assign(&mut ci, &T::and(v1, v2));

            *v1 = xor_val;
            ci
        })
}

pub(crate) fn add_vectors_assign<T: BooleanType<C>, C>(
    ctx: &Arc<C>,
    lst1: &mut [T],
    lst2: Vec<&T>,
) {
    let c = lst1
        .iter_mut()
        .take(lst2.len() - 1)
        .zip(lst2.iter())
        .fold::<T, _>(T::from_ctx(false, ctx), |c, (v1, v2)| {
            let mut xor_val = T::xor(v1, v2);
            let mut ci = T::and(&c, &xor_val);

            T::xor_assign(&mut xor_val, &c);
            T::or_assign(&mut ci, &T::and(v1, v2));

            *v1 = xor_val;
            ci
        });
    let v1 = lst1.last_mut().unwrap();
    let v2 = lst2.last().unwrap();

    T::xor_assign(v1, v2);
    T::xor_assign(v1, &c);
}

pub(crate) fn sub_vectors_assign<T: BooleanType<C>, C>(
    ctx: &Arc<C>,
    lst1: &mut [T],
    lst2: Vec<&T>,
) {
    let c = lst1
        .iter_mut()
        .take(lst2.len() - 1)
        .zip(lst2.iter())
        .fold::<T, _>(T::from_ctx(false, ctx), |c, (v1, v2)| {
            let mut not_a = v1.clone();
            not_a.not_assign();
            not_a.and_assign(*v2);

            T::xor_assign(v1, v2);

            let mut v1_v2 = v1.clone();

            v1.xor_assign(&c);

            v1_v2.not_assign();
            v1_v2.and_assign(&c);

            not_a.or_assign(&v1_v2);
            not_a
        });

    let v1 = lst1.last_mut().unwrap();
    let v2 = lst2.last().unwrap();

    T::xor_assign(v1, *v2);
    v1.xor_assign(&c);
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    AddAssign<&FixedPointNumber<SIZE, PRECISION, T, T1>>
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn add_assign(&mut self, rhs: &FixedPointNumber<SIZE, PRECISION, T, T1>) {
        let b = Arc::make_mut(&mut self.bits);

        add_vectors_assign(&self.context, b, rhs.bits.iter().collect());
    }
}
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    Add<&FixedPointNumber<SIZE, PRECISION, T, T1>> for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn add(mut self, rhs: &FixedPointNumber<SIZE, PRECISION, T, T1>) -> Self::Output {
        let b = Arc::make_mut(&mut self.bits);
        add_vectors_assign(&self.context, b, rhs.bits.iter().collect());
        self
    }
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    Sub<&FixedPointNumber<SIZE, PRECISION, T, T1>> for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn sub(mut self, rhs: &FixedPointNumber<SIZE, PRECISION, T, T1>) -> Self::Output {
        let b = Arc::make_mut(&mut self.bits);
        sub_vectors_assign(&self.context, b, rhs.bits.iter().collect());
        self
    }
}
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    SubAssign<&FixedPointNumber<SIZE, PRECISION, T, T1>>
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn sub_assign(&mut self, rhs: &FixedPointNumber<SIZE, PRECISION, T, T1>) {
        let b = Arc::make_mut(&mut self.bits);
        sub_vectors_assign(&self.context, b, rhs.bits.iter().collect());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{FixedPointNumber, FromWithContext};

    #[test]
    fn f32_addition() {
        let ctx = Arc::new(());

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res = f1 + &f2;
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 + v2, "f32_adition {} + {} ", v1, v2);
        };

        calc(0.0, 0.0);
        calc(-0.0, 0.0);
        calc(-0.0, 1.0);
        calc(1.0, 1.0);
        calc(3.5, 1.0);
        calc(-1.0, 1.0);
        calc(-2.0, 1.5);
        calc(4.0, -3.0);
        calc(-2.5, -3.0);
        calc(-2.5, 2.0);
        calc(-0.5, -1.0);
        calc(8.0, 1.0);
    }

    #[test]
    fn f32_subtraction() {
        let ctx = Arc::new(());

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res = f1 - &f2;
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 - v2, "f32_subtraction {} - {} ", v1, v2);
        };

        calc(0.0, 0.0);
        calc(-0.0, 0.0);
        calc(-0.0, 1.0);
        calc(1.0, 1.0);
        calc(3.5, 1.0);
        calc(-1.0, 1.0);
        calc(-2.0, 1.5);
        calc(4.0, -3.0);
        calc(-2.5, -3.0);
        calc(-0.5, -1.0);
        calc(8.0, 1.0);
    }
}
