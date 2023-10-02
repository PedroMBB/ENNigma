use std::{ops::Not, sync::Arc};

use crate::{BooleanType, FixedPointNumber};

pub(crate) fn twoscomplement_vectors<T: BooleanType<C>, C>(ctx: &Arc<C>, lst: &mut [T]) {
    let len = lst.len();

    let b = lst
        .iter_mut()
        .take(len - 1)
        .fold::<T, _>(T::from_ctx(true, ctx), |mut b, a| {
            T::not_assign(a);
            let a_origin = a.clone();

            T::xor_assign(a, &b);
            T::and_assign(&mut b, &a_origin);
            b
        });
    let a = lst.last_mut().unwrap();

    T::not_assign(a);
    T::xor_assign(a, &b);
}

pub(crate) fn twoscomplement_switch<T: BooleanType<C>, C>(ctx: &Arc<C>, sign: &T, lst: &mut [T]) {
    let mut b = T::from_ctx(true, ctx);
    let len = lst.len();

    lst.iter_mut().take(len - 1).for_each(|a| {
        let original_a = a.clone();
        T::xnor_assign(a, &b);
        let mut not_a = a.clone();
        not_a.not_assign();
        T::and_assign(&mut b, &not_a);

        *a = sign.mux(&a, &original_a);
    });

    let a = lst.last_mut().unwrap();
    let original_a = a.clone();
    T::xnor_assign(a, &b);

    *a = sign.mux(&a, &original_a);
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Not
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn not(mut self) -> Self::Output {
        let b = Arc::make_mut(&mut self.bits);
        twoscomplement_vectors(&self.context, b);
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{FixedPointNumber, FromWithContext};

    #[test]
    fn f32_2scomplement() {
        let ctx = Arc::new(());
        let calc = |v1: f32| {
            let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);

            let enc_res = !f1;
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 * -1.0, "f32_2scomplement {} ", v1);
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
