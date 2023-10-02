use std::{fmt::Display, marker::PhantomData, sync::Arc};

use crate::{
    iterator_to_sized_arc, AddContext, BooleanType, DefaultWithContext, FixedPointNumber,
    FixedPointNumberNoContext, FromWithContext,
};

impl<
        const SIZE: usize,
        const PRECISION: usize,
        T: BooleanType<C> + AddContext<C, FromType = T1>,
        C,
        T1,
    > AddContext<C> for FixedPointNumber<SIZE, PRECISION, T, C>
{
    type FromType = FixedPointNumberNoContext<SIZE, PRECISION, T1>;

    fn add_context(t: &Self::FromType, ctx: &Arc<C>) -> Self {
        Self {
            bits: iterator_to_sized_arc(t.bits.iter().map(|v| AddContext::add_context(v, ctx))),
            precision: t.precision,
            context: Arc::clone(ctx),
        }
    }
}

// Switch Context
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    FixedPointNumber<SIZE, PRECISION, T, C>
{
    pub fn switch_context(mut self, context: &Arc<C>) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let bits = Arc::make_mut(&mut self.bits);
        bits.iter_mut().for_each(|f| *f = f.switch_context(context));
        self.context = Arc::clone(context);
        self
    }
}

// Clone
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C> Clone
    for FixedPointNumber<SIZE, PRECISION, T, C>
{
    fn clone(&self) -> Self {
        Self {
            bits: Arc::clone(&self.bits),
            precision: PhantomData::default(),
            context: Arc::clone(&self.context),
        }
    }
}

// Default
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C> DefaultWithContext<C>
    for FixedPointNumber<SIZE, PRECISION, T, C>
{
    fn default_ctx(ctx: &Arc<C>) -> Self {
        Self {
            bits: iterator_to_sized_arc((0..SIZE).map(|_| T::from_ctx(false, ctx))),
            precision: PhantomData::default(),
            context: ctx.clone(),
        }
    }
}

impl<const S: usize, const P: usize, T: BooleanType<C>, C> Display for FixedPointNumber<S, P, T, C>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v: f32 = self.clone().into();
        write!(f, "{}", v)
        // write!(
        //     f,
        //     "FixedPointNumber: [{}]",
        //     self.bits
        //         .iter()
        //         .map(|v| v.to_string())
        //         .collect::<Vec<_>>()
        //         .join(",")
        // )
    }
}

// Update Size
impl<const S1: usize, const P1: usize, T: BooleanType<C>, C> FixedPointNumber<S1, P1, T, C> {
    pub fn update_size<const S2: usize, const P2: usize>(&self) -> FixedPointNumber<S2, P2, T, C> {
        let mut lst = Vec::with_capacity(S2);
        let msb = self.msb();

        // Fill missing precision
        for _ in 0..(P2 as isize - P1 as isize).max(0) {
            lst.push(FromWithContext::from_ctx(false, &self.context));
        }
        // Fill existing precision
        lst.extend(
            self.bits
                .iter()
                .take(P1)
                .rev()
                .take(P2.min(P1))
                .rev()
                .cloned(),
        );

        let np1 = S1 as isize - P1 as isize - 1;
        let np2 = S2 as isize - P2 as isize - 1;

        // Fill existing size
        lst.extend(
            self.bits
                .iter()
                .skip(P1)
                .take(np2.min(np1) as usize)
                .cloned(),
        );

        // Fill remaining size
        for _ in 0..(np2 - np1).max(0) {
            lst.push(msb.clone());
        }
        lst.push(msb.clone());

        FixedPointNumber::from_bits(lst, &self.context)
    }
}

// Switch Context
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    FixedPointNumber<SIZE, PRECISION, T, C>
{
    pub fn optimize_integer_size(mut self, int_bits: usize, consider_positive: bool) -> Self {
        let b = Arc::make_mut(&mut self.bits);
        let fill = T::from_ctx(!consider_positive, &self.context);

        let start = PRECISION + int_bits;
        if start <= SIZE {
            for i in start..SIZE {
                b[i] = fill.clone();
            }
        }

        self
    }
    pub fn optimize_integer_size_ref(&mut self, int_bits: usize, consider_positive: bool) {
        let b = Arc::make_mut(&mut self.bits);
        let fill = T::from_ctx(!consider_positive, &self.context);

        let start = PRECISION + int_bits;
        if start <= SIZE {
            for i in start..SIZE {
                b[i] = fill.clone();
            }
        }
    }
}

// Others
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    FixedPointNumber<SIZE, PRECISION, T, C>
{
    pub fn msb(&self) -> &T {
        self.bits.last().expect("Should have at least one bit")
    }
    pub fn mux(condition: &T, mut yes: Self, no: Self) -> Self {
        let b = Arc::make_mut(&mut yes.bits);

        b.iter_mut()
            .zip(no.bits.iter())
            .for_each(|(yes, no)| *yes = condition.mux(yes, no));

        yes
    }
    pub fn mux_number(mut self, yes: &Self, no: &Self) -> Self {
        let bit = self.bits[PRECISION].clone();
        let b = Arc::make_mut(&mut self.bits);

        b.iter_mut()
            .zip(yes.bits.iter().zip(no.bits.iter()))
            .for_each(|(c, (y, n))| *c = bit.mux(y, n));

        self
    }
    pub fn mux_number_ref(&self, mut yes: Self, no: &Self) -> Self {
        let bit = self.bits[PRECISION].clone();
        let b = Arc::make_mut(&mut yes.bits);

        b.iter_mut()
            .zip(no.bits.iter())
            .for_each(|(y, n)| *y = bit.mux(y, n));

        yes
    }
    fn new(val: usize, context: Arc<C>) -> Self {
        let lst: Vec<bool> = (0..usize::BITS)
            .into_iter()
            .fold(
                (1, Vec::with_capacity(usize::BITS as usize)),
                |(mult, mut acc), _| {
                    acc.push(match val & mult {
                        0 => false,
                        _ => true,
                    });
                    (mult << 1, acc)
                },
            )
            .1;

        let mut bits = Vec::with_capacity(SIZE);
        for i in 0..SIZE {
            bits.push(T::from_ctx(lst[i], &context))
        }

        Self {
            bits: iterator_to_sized_arc(bits.into_iter()),
            precision: PhantomData::default(),
            context,
        }
    }
    pub(crate) fn from_bits(bits: Vec<T>, context: &Arc<C>) -> Self {
        Self {
            bits: iterator_to_sized_arc(bits.into_iter()),
            precision: PhantomData::default(),
            context: Arc::clone(context),
        }
    }
    pub(crate) fn from_iter<I: Iterator<Item = T>>(iter: I, context: &Arc<C>) -> Self {
        Self {
            bits: iterator_to_sized_arc(iter),
            precision: PhantomData::default(),
            context: Arc::clone(context),
        }
    }
}

/*
   i8
*/
impl<const PRECISION: usize, T: BooleanType<C>, C> FromWithContext<i8, C>
    for FixedPointNumber<8, PRECISION, T, C>
{
    fn from_ctx(value: i8, ctx: &Arc<C>) -> Self {
        let bytes = value.to_le_bytes();

        Self::new(bytes[0].into(), Arc::clone(ctx))
    }
}
impl<const PRECISION: usize, T: BooleanType<T1>, T1> Into<i8>
    for FixedPointNumber<8, PRECISION, T, T1>
{
    fn into(self) -> i8 {
        self.bits
            .iter()
            .map::<bool, _>(|v| v.clone().into())
            .rev()
            .fold(0_i8, |p, v| {
                (p << 1)
                    | match v {
                        false => 0,
                        true => 1,
                    }
            })
    }
}

/*
   f32
*/
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C> FromWithContext<f32, C>
    for FixedPointNumber<SIZE, PRECISION, T, C>
{
    fn from_ctx(value: f32, ctx: &Arc<C>) -> Self {
        let pow = 2_isize.pow(
            (PRECISION)
                .try_into()
                .expect("Could not calculate the power given the precision and the size"),
        ) as f64;

        let n: isize = (pow * (value as f64)) as isize;
        Self::new(n as usize, Arc::clone(ctx))
    }
}
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Into<f32>
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn into(self) -> f32 {
        let size: i32 = SIZE.try_into().expect("Should convert to i32");
        let press: i32 = PRECISION.try_into().expect("Should convert to i32");

        let sift = size - press - 1;

        self.bits
            .iter()
            .map(|v| match v.clone().into() {
                false => 0.0,
                true => 1.0,
            })
            .rev()
            .enumerate()
            .fold(0.0, |prev, (i, curr)| {
                let i: i32 = i.try_into().expect("Should convert to i32");
                if i == 0 {
                    return curr * -2.0_f32.powi(sift);
                }
                prev + curr * 2_f32.powi(sift - i)
            })
    }
}

/*
   usize
*/
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<T1>, T1> Into<usize>
    for FixedPointNumber<SIZE, PRECISION, T, T1>
{
    fn into(self) -> usize {
        self.bits
            .iter()
            .map::<bool, _>(|v| v.clone().into())
            .rev()
            .take(usize::BITS as usize)
            .fold(0_usize, |p, v| {
                (p << 1)
                    | match v {
                        false => 0,
                        true => 1,
                    }
            })
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use super::FixedPointNumber;
    use crate::FromWithContext;

    #[test]
    fn test() {
        let ctx = Arc::new(());

        let f1: FixedPointNumber<8, 2, bool, ()> = FixedPointNumber::from_ctx(1.0, &ctx);
        println!("{:?}", f1);
    }

    #[test]
    fn i8_conversion() {
        let pow = 2_i32.pow(3) as f32;

        let n1: i8 = (pow * 0.0) as i8;
        let n2: i8 = (pow * -1.5) as i8;
        let n3: i8 = (pow * 2.5) as i8;
        let n4: i8 = (pow * 0.5) as i8;
        let n5: i8 = (pow * 2.0) as i8;

        let ctx = Arc::new(());

        let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n1, &ctx);
        let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n2, &ctx);
        let f3: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n3, &ctx);
        let f4: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n4, &ctx);
        let f5: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n5, &ctx);

        let i1: i8 = f1.into();
        let i2: i8 = f2.into();
        let i3: i8 = f3.into();
        let i4: i8 = f4.into();
        let i5: i8 = f5.into();

        assert_eq!(n1, i1);
        assert_eq!(n2, i2);
        assert_eq!(n3, i3);
        assert_eq!(n4, i4);
        assert_eq!(n5, i5);
    }

    #[test]
    fn f32_conversion() {
        let n1: f32 = 1.0;
        let n2: f32 = -1.5;
        let n3: f32 = 2.5;
        let n4: f32 = 0.5;
        let n5: f32 = 2.0;
        let n6: f32 = 3.0;
        let n7: f32 = 125.27344;

        let ctx = Arc::new(());

        let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n1, &ctx);
        let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n2, &ctx);
        let f2b: FixedPointNumber<8, 4, bool, ()> = FixedPointNumber::from_ctx(n2, &ctx);
        let f2c: FixedPointNumber<16, 8, bool, ()> = FixedPointNumber::from_ctx(n2, &ctx);
        let f3: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n3, &ctx);
        let f4: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n4, &ctx);
        let f5: FixedPointNumber<5, 1, bool, ()> = FixedPointNumber::from_ctx(n5, &ctx);
        let f6: FixedPointNumber<5, 1, bool, ()> = FixedPointNumber::from_ctx(n6, &ctx);
        let f7: FixedPointNumber<32, 8, bool, ()> = FixedPointNumber::from_ctx(n7, &ctx);

        let i1: f32 = f1.into();
        let i2: f32 = f2.into();
        let i2b: f32 = f2b.into();
        let i2c: f32 = f2c.into();
        let i3: f32 = f3.into();
        let i4: f32 = f4.into();
        let i5: f32 = f5.into();
        let i6: f32 = f6.into();
        let i7: f32 = f7.into();

        assert_eq!(n1, i1);
        assert_eq!(n2, i2);
        assert_eq!(n2, i2b);
        assert_eq!(n2, i2c);
        assert_eq!(n3, i3);
        assert_eq!(n4, i4);
        assert_eq!(n5, i5);
        assert_eq!(n6, i6);
        assert_eq!(n7, i7);
    }

    #[test]
    fn f32_conversion_ignoreoverflow() {
        let n1: f32 = 1.45515616674457554;
        let n2: f32 = -1.15563566546474;

        let ctx = Arc::new(());

        let f1: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n1, &ctx);
        let f2: FixedPointNumber<8, 3, bool, ()> = FixedPointNumber::from_ctx(n2, &ctx);

        let i1: f32 = f1.into();
        let i2: f32 = f2.into();

        assert_eq!(i1, 1.375);
        assert_eq!(i2, -1.125);
    }

    #[test]
    fn change_precision() {
        fn check<const S1: usize, const P1: usize, const S2: usize, const P2: usize>(v: f32) {
            let ctx = Arc::new(());

            let f1: FixedPointNumber<S1, P1, bool, ()> = FixedPointNumber::from_ctx(v, &ctx);
            let f2: FixedPointNumber<S2, P2, bool, ()> = f1.update_size();

            let i1: f32 = f1.into();
            let i2: f32 = f2.into();

            assert_eq!(i1, i2);
        }

        check::<4, 2, 4, 2>(2.0);
        check::<4, 2, 8, 4>(0.0);
        check::<4, 2, 8, 4>(1.0);
        check::<6, 3, 8, 4>(2.25);
        check::<8, 3, 8, 4>(2.5);
        check::<8, 3, 8, 4>(-1.5);
        check::<4, 2, 8, 4>(-1.0);

        check::<16, 8, 8, 2>(2.0);
        check::<10, 4, 8, 4>(0.0);
        check::<16, 7, 16, 2>(1.0);
        check::<12, 4, 12, 8>(2.25);
        check::<8, 1, 8, 4>(2.5);
        check::<24, 10, 10, 4>(-1.5);
        check::<8, 4, 4, 2>(-1.0);
    }

    #[test]
    fn optimize_integer_size() {
        fn check<const S1: usize, const P1: usize>(v: f32, size: usize, positive: bool, exp: f32) {
            let ctx = Arc::new(());

            let f1: FixedPointNumber<S1, P1, bool, ()> = FixedPointNumber::from_ctx(v, &ctx);
            let f2 = f1.optimize_integer_size(size, positive);

            let i2: f32 = f2.into();

            assert_eq!(exp, i2);
        }

        check::<16, 8>(2.5, 2, true, 2.5);
        check::<16, 8>(-2.5, 2, false, -2.5);
        check::<10, 4>(-2.5, 10, false, -2.5);
        check::<10, 4>(1.5, 1, true, 1.5);
    }
}
