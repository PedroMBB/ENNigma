use super::twoscomp::twoscomplement_switch;
use crate::{BooleanType, FixedPointNumber};
use std::ops::Mul;
use std::sync::Arc;

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    Mul<&FixedPointNumber<SIZE, PRECISION, T, T1>> for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn mul(mut self, rhs: &FixedPointNumber<SIZE, PRECISION, T, T1>) -> Self::Output {
        let ctx = Arc::clone(&self.context);
        let false_val = T::from_ctx(false, &ctx);

        let sign = T::xor(self.msb(), rhs.msb());

        let a_sign = self.msb().clone();
        let abs_a = Arc::make_mut(&mut self.bits);
        twoscomplement_switch(&self.context, &a_sign, abs_a);

        let mut b_arc = Arc::clone(&rhs.bits);
        let b_sign = rhs.msb().clone();
        let abs_b = Arc::make_mut(&mut b_arc);
        twoscomplement_switch(&self.context, &b_sign, abs_b);

        let mut m: Vec<Option<T>> = (0..(SIZE - 1)).map(|_| None).collect();

        abs_b
            .iter()
            .take(SIZE - 1)
            .enumerate()
            .for_each(|(i, b_val)| {
                let (prepend, remaining) = match i > PRECISION {
                    true => (i - PRECISION, 0),
                    false => (0, PRECISION - i),
                };

                m.iter_mut()
                    .skip(prepend)
                    .zip(abs_a.iter().skip(remaining).take(SIZE - prepend))
                    .fold(false_val.clone(), |c, (m_val, a_val)| {
                        let res = T::and(a_val, b_val);

                        match m_val.as_mut() {
                            None => {
                                *m_val = Some(res);
                                false_val.clone()
                            }
                            Some(m_val) => {
                                let v1 = m_val;
                                let v2 = &res;
                                let mut xor_val = T::xor(v1, v2);
                                let mut ci = T::and(&c, &xor_val);

                                T::xor_assign(&mut xor_val, &c);
                                T::or_assign(&mut ci, &T::and(v1, v2));

                                *v1 = xor_val;
                                ci
                            }
                        }
                    });
            });
        let mut m: Vec<T> = m.into_iter().collect::<Option<_>>().unwrap();
        m.push(false_val.clone());

        twoscomplement_switch(&ctx, &sign, &mut m);

        Self::Output::from_bits(m, &ctx)
    }
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    Mul<&FixedPointNumber<SIZE, PRECISION, T, T1>> for &FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;
    fn mul(self, rhs: &FixedPointNumber<SIZE, PRECISION, T, T1>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1>
    Mul<(&T, &FixedPointNumber<SIZE, PRECISION, T, T1>, &T, &T)>
    for &FixedPointNumber<SIZE, PRECISION, T, T1>
{
    type Output = FixedPointNumber<SIZE, PRECISION, T, T1>;

    fn mul(
        self,
        (self_msb, abs_b, rhs_msb, false_val): (
            &T,
            &FixedPointNumber<SIZE, PRECISION, T, T1>,
            &T,
            &T,
        ),
    ) -> Self::Output {
        let sign = T::xor(self_msb, rhs_msb);
        let abs_a = self;

        let mut m: Vec<Option<T>> = (0..(SIZE - 1)).map(|_| None).collect();

        abs_b
            .bits
            .iter()
            .take(SIZE - 1)
            .enumerate()
            .for_each(|(i, b_val)| {
                let (prepend, remaining) = match i > PRECISION {
                    true => (i - PRECISION, 0),
                    false => (0, PRECISION - i),
                };

                m.iter_mut()
                    .skip(prepend)
                    .zip(abs_a.bits.iter().skip(remaining).take(SIZE - prepend))
                    .fold(false_val.clone(), |c, (m_val, a_val)| {
                        let res = T::and(a_val, b_val);

                        match m_val.as_mut() {
                            None => {
                                *m_val = Some(res);
                                false_val.clone()
                            }
                            Some(m_val) => {
                                let v1 = m_val;
                                let v2 = &res;
                                let mut xor_val = T::xor(v1, v2);
                                let mut ci = T::and(&c, &xor_val);

                                T::xor_assign(&mut xor_val, &c);
                                T::or_assign(&mut ci, &T::and(v1, v2));

                                *v1 = xor_val;
                                ci
                            }
                        }
                    });
            });
        let mut m: Vec<T> = m.into_iter().collect::<Option<_>>().unwrap();
        m.push(false_val.clone());

        twoscomplement_switch(&self.context, &sign, &mut m);

        Self::Output::from_bits(m, &self.context)
    }
}

pub use karatsuba::karatsuba_adapt;
mod karatsuba {
    use std::sync::Arc;

    use crate::{
        fixedpoint::operations::twoscomp::twoscomplement_vectors, Abs, BooleanType,
        FixedPointNumber,
    };

    pub fn karatsuba_adapt<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>(
        lhs: &FixedPointNumber<SIZE, PRECISION, T, C>,
        rhs: &FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let ctx = lhs.context.clone();
        let false_val = T::from_ctx(false, &ctx);

        let sign = T::xor(lhs.msb(), rhs.msb());

        let abs_a = lhs.clone().abs();
        let abs_b = rhs.clone().abs();

        let l_bits: Vec<_> = abs_a
            .bits
            .iter()
            .take(SIZE - 1)
            .map(|v| Some(v.clone()))
            .collect();
        let r_bits: Vec<_> = abs_b
            .bits
            .iter()
            .take(SIZE - 1)
            .map(|v| Some(v.clone()))
            .collect();

        let mut stack: Vec<_> = (0..(lhs.bits.len() * 11)).map(|_| None).collect();
        let mut dest: Vec<_> = (0..(lhs.bits.len() * 2 - 2)).map(|_| None).collect();

        karatsuba(&l_bits, &r_bits, &mut dest, &mut stack, &ctx, 0);

        let mut m: Vec<T> = dest
            .into_iter()
            .skip(PRECISION)
            .take(SIZE - 1)
            .map(|v| v.unwrap_or(false_val.clone()))
            .collect();

        m.push(false_val.clone());

        let mut not_m = m.clone();
        twoscomplement_vectors(&ctx, &mut not_m);

        let new_bits: Vec<_> = m
            .iter()
            .zip(not_m.iter())
            .map(|(a, b)| sign.mux(b, a))
            .collect();

        FixedPointNumber::from_bits(new_bits, &ctx)
    }

    fn karatsuba<T: BooleanType<T1>, T1>(
        lhs: &[Option<T>],
        rhs: &[Option<T>],
        dest: &mut [Option<T>],
        stack: &mut [Option<T>],
        ctx: &Arc<T1>,
        i: usize,
    ) {
        let size = lhs.len();

        if size == 0 {
            return;
        }
        if size == 1 {
            dest[0] = match (lhs[0].as_ref(), rhs[0].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            };
            dest[1] = None;
            return;
        }
        if size == 2 {
            dest[0] = match (lhs[1].as_ref(), rhs[0].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            }; // I3
            dest[2] = match (lhs[1].as_ref(), rhs[1].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            }; // I4

            stack[0] = match (lhs[0].as_ref(), rhs[1].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            }; // I1

            dest[1] = match (stack[0].as_ref(), dest[0].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            }; // I6

            dest[3] = match (dest[2].as_ref(), dest[1].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            }; // dest 3
            dest[2] = match (dest[2].as_ref(), dest[1].as_ref()) {
                (Some(a), Some(b)) => Some(a.xor(b)),
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                _ => None,
            }; // dest 2

            dest[1] = match (stack[0].as_ref(), dest[0].as_ref()) {
                (Some(a), Some(b)) => Some(a.xor(b)),
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                _ => None,
            }; // I6

            dest[0] = match (lhs[0].as_ref(), rhs[0].as_ref()) {
                (Some(a), Some(b)) => Some(a.and_ref(b)),
                _ => None,
            }; // dest 3

            return;
        }

        let new_size = (size as f32 / 2.0).ceil() as usize;

        let (a0, a1) = lhs.split_at(new_size);
        let (b0, b1) = rhs.split_at(new_size);

        let (p1, stack) = stack.split_at_mut(new_size * 2);
        let (p2, stack) = stack.split_at_mut(new_size * 2);
        let (p3, stack) = stack.split_at_mut(new_size * 4);

        {
            let (p1, p1_rem) = p1.split_at_mut(new_size + 1);
            let (p2, p2_rem) = p2.split_at_mut(new_size + 1);

            for i in 0..(new_size - 1) {
                p1_rem[i] = None;
                p2_rem[i] = None;
            }

            karatsuba_add(ctx, a0, a1, p1);
            karatsuba_add(ctx, b0, b1, p2);
            karatsuba(p1, p2, p3, stack, ctx, i + 1); // e1 = x1 * y1
        }

        karatsuba(a1, b1, p1, stack, ctx, i + 1); // a = xH * yH
        karatsuba(a0, b0, p2, stack, ctx, i + 1); // d = xL * yL

        let (v2, stack) = stack.split_at_mut(new_size * 4);
        let (v1, stack) = stack.split_at_mut(new_size * 4);
        let (v3, _stack) = stack.split_at_mut(new_size * 4);

        // v2 = P3 - (P2 + P1)
        karatsuba_add(ctx, p2, p1, v2);
        for i in (new_size * 2)..(new_size * 4) {
            v2[i] = None;
        }
        karatsuba_sub_assign_last(ctx, p3, v2);

        // v3 = v2 * b^(size / 2)
        for i in 0..new_size {
            v3[i] = None;
            v3[i + new_size] = v2[i].clone();
            v3[i + new_size + new_size] = v2[i + new_size].clone();
            v3[i + new_size + new_size + new_size] = None;
        }

        // v1 = a * b^size
        for i in 0..(new_size * 2) {
            v1[i] = None;
            v1[i + (new_size * 2)] = p1[i].clone();
        }

        // dest = (v1 + v3)
        karatsuba_add(ctx, v1, v3, dest);

        if i == 0 {
            println!("LHS {:?}", lhs);
            println!("RHS {:?}", rhs);
            println!("A0 {:?}, A1 {:?}", a0, a1);
            println!("B0 {:?}, B1 {:?}", b0, b1);
            println!("P1 {:?}", p1);
            println!("P2 {:?}", p2);
            println!("P3 {:?}", p3);

            println!("P3 - P1 - P2 {:?}", v2);
            println!("(P3 - P1 - P2) * 2^(size / 2) {:?}", v3);
            println!("P1 * 2^(size) {:?}", v1);
            println!("P1 * 2^(size) + (P3 - P1 - P2) {:?}", dest);
        }

        // res = (v1 + v3) + p1
        for i in 0..new_size {
            v1[i] = p1[i].clone();
            v1[i + new_size] = None;
            v3[i] = dest[i].clone();
            v3[i + new_size] = dest[i + new_size].clone();
        }

        karatsuba_add(ctx, v1, v3, dest);
    }

    fn karatsuba_add<T: BooleanType<C>, C>(
        ctx: &Arc<C>,
        lst1: &[Option<T>],
        lst2: &[Option<T>],
        dest: &mut [Option<T>],
    ) {
        let false_val = T::from_ctx(false, &ctx);

        let c = lst1
            .into_iter()
            .take(lst2.len() - 1)
            .zip(lst2.into_iter())
            .zip(dest.iter_mut())
            .fold::<T, _>(T::from_ctx(false, ctx), |c, ((v1, v2), dest)| {
                let v1 = v1.as_ref().unwrap_or(&false_val);
                let v2 = v2.as_ref().unwrap_or(&false_val);

                let mut xor_val = T::xor(v1, v2);
                let mut ci = T::and(&c, &xor_val);

                T::xor_assign(&mut xor_val, &c);
                T::or_assign(&mut ci, &T::and(v1, v2));

                *dest = Some(xor_val);
                ci
            });
        let dest = dest.last_mut().unwrap();
        let v1 = lst1.last().unwrap().as_ref().unwrap_or(&false_val);
        let v2 = lst2.last().unwrap().as_ref().unwrap_or(&false_val);

        let mut res = T::xor(v1, v2);
        T::xor_assign(&mut res, &c);
        *dest = Some(res);
    }

    fn karatsuba_sub_assign_last<T: BooleanType<C>, C>(
        ctx: &Arc<C>,
        lst1: &[Option<T>],
        lst2: &mut [Option<T>],
    ) {
        let false_val = T::from_ctx(false, &ctx);

        let c = lst1
            .into_iter()
            .take(lst2.len() - 1)
            .zip(lst2.iter_mut())
            .fold::<T, _>(T::from_ctx(false, ctx), |c, (v1, v2)| {
                let (res, not_a) = {
                    let v1 = v1.as_ref().unwrap_or(&false_val);
                    let v2 = v2.as_ref().unwrap_or(&false_val);

                    let mut not_a = v1.clone();
                    not_a.not_assign();
                    not_a.and_assign(v2);

                    let mut not_v1 = T::xor(v1, v2);

                    let mut v1_v2 = not_v1.clone();

                    not_v1.xor_assign(&c);

                    v1_v2.not_assign();
                    v1_v2.and_assign(&c);

                    not_a.or_assign(&v1_v2);
                    (not_v1, not_a)
                };
                *v2 = Some(res);
                not_a
            });
        let v1 = lst1.last().unwrap().as_ref().unwrap_or(&false_val);
        let v2 = lst2.last_mut().unwrap().as_ref().unwrap_or(&false_val);

        let mut res = T::xor(v1, v2);
        res.xor_assign(&c);

        let v2 = lst2.last_mut().unwrap();
        *v2 = Some(res);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::karatsuba::karatsuba_adapt;
    use crate::{FixedPointNumber, FromWithContext};

    #[test]
    fn f32_mult() {
        let ctx = Arc::new(());

        // let calc = |v1: f32, v2: f32| {
        //     let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
        //     let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

        //     let enc_res = &f1 * &f2;
        //     let res: f32 = enc_res.into();

        //     assert_eq!(res, v1 * v2, "f32_mult {} * {} ", v1, v2);
        // };

        let calc = |v1: f32, v2: f32| {
            let f1: FixedPointNumber<32, 12, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: FixedPointNumber<32, 12, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res = f1 * &f2;
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 * v2, "f32_mult {} * {} ", v1, v2);
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
        calc(-0.5, 0.5);
        calc(8.0, 1.0);
    }

    #[test]
    fn f32_bit_mult() {
        let ctx = Arc::new(());

        // let calc = |v1: f32, v2: f32| {
        //     let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v1, &ctx);
        //     let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(v2, &ctx);

        //     let enc_res = &f1 * &f2;
        //     let res: f32 = enc_res.into();

        //     assert_eq!(res, v1 * v2, "f32_mult {} * {} ", v1, v2);
        // };

        type N = FixedPointNumber<32, 4, bool, ()>;

        let calc = |v1: f32, v2: f32| {
            let f1: N = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: N = FixedPointNumber::from_ctx(v2, &ctx);
            let exp: N = FixedPointNumber::from_ctx(v1 * v2, &ctx);

            let enc_res: N = karatsuba_adapt(&f1, &f2);
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 * v2, "f32_mult {} * {} ", v1, v2);
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
        calc(-0.5, 0.5);
        calc(8.0, 1.0);
        calc(8.0, 4.0);
        calc(8.0, 8.0);

        type N1 = FixedPointNumber<16, 8, bool, ()>;

        let calc = |v1: f32, v2: f32| {
            let f1: N1 = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: N1 = FixedPointNumber::from_ctx(v2, &ctx);
            let exp: N1 = FixedPointNumber::from_ctx(v1 * v2, &ctx);

            let enc_res: N1 = karatsuba_adapt(&f1, &f2);
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 * v2, "f32_mult {} * {} ", v1, v2);
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
        calc(-0.5, 0.5);
        calc(8.0, 1.0);
        calc(8.0, 4.0);
        calc(8.0, 8.0);
    }

    #[test]
    fn f32_karatsuba() {
        fn calc<const SIZE: usize, const PRECISION: usize>(v1: f32, v2: f32) {
            let ctx = Arc::new(());

            type N<const S: usize, const P: usize> = FixedPointNumber<S, P, bool, ()>;

            let f1: N<SIZE, PRECISION> = FixedPointNumber::from_ctx(v1, &ctx);
            let f2: N<SIZE, PRECISION> = FixedPointNumber::from_ctx(v2, &ctx);

            let enc_res: N<SIZE, PRECISION> = karatsuba_adapt(&f1, &f2);
            let res: f32 = enc_res.into();

            assert_eq!(res, v1 * v2, "f32_mult {} * {} ", v1, v2);
        }

        // calc::<2, 0>(0.0, 0.0);
        // calc::<2, 0>(1.0, 0.0);
        // calc::<2, 0>(1.0, 1.0);
        // calc::<2, 0>(0.0, 0.0);

        // calc::<3, 0>(1.0, 1.0);
        // calc::<3, 0>(2.0, 1.0);
        calc::<7, 0>(3.0, 3.0);
    }
}
