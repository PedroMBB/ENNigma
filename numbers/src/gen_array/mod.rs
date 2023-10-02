use array_macro::array;
use serde::Serialize;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, Sub, SubAssign},
    sync::Arc,
};

use crate::{Abs, BooleanType, DefaultWithContext, FixedPointNumber, FromWithContext};

mod gen1darray;
pub use gen1darray::Gen1DArray;

mod noctx;
pub use noctx::Gen2DArrayNoContext;

pub struct Gen2DArray<T, C, const ROWS: usize, const COLS: usize> {
    contents: Box<[T]>,
    context: Arc<C>,
    rows: PhantomData<[T; ROWS]>,
    cols: PhantomData<[T; COLS]>,
}

impl<T: Serialize, const ROWS: usize, const COLS: usize, C> Serialize
    for Gen2DArray<T, C, ROWS, COLS>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let v: &[T] = self.contents.as_ref();

        v.serialize(serializer)
    }
}

#[cfg(feature = "rand")]
impl<T: Clone, const ROWS: usize, const COLS: usize, C> Gen2DArray<T, C, ROWS, COLS> {
    pub fn random(
        rand: &mut dyn rand::RngCore,
        generator: impl Fn(&mut dyn rand::RngCore) -> T,
        ctx: &Arc<C>,
    ) -> Self {
        let size = COLS * ROWS;

        Self {
            contents: (0..size).map(|_| generator(rand)).collect(),
            context: Arc::clone(ctx),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T: Clone, const ROWS: usize, const COLS: usize, C> Gen2DArray<T, C, ROWS, COLS> {
    pub fn as_array(&self) -> [[T; COLS]; ROWS] {
        array![i => array![j =>self.contents[i * COLS + j].clone(); COLS]; ROWS]
    }
}
impl<T: Clone, const ROWS: usize, const COLS: usize, C> Gen2DArray<T, C, ROWS, COLS> {
    pub fn as_slice(&self) -> [[&T; COLS]; ROWS] {
        array![i => array![j => &self.contents[i * COLS + j]; COLS]; ROWS]
    }
}
impl<T: Debug, C, const N1: usize, const N2: usize> Debug for Gen2DArray<T, C, N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.contents))
    }
}
impl<T: Clone, C, const N1: usize, const N2: usize> Clone for Gen2DArray<T, C, N1, N2> {
    fn clone(&self) -> Self {
        Self {
            contents: self.contents.clone(),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T, C, const ROWS: usize, const COLS: usize> Gen2DArray<T, C, ROWS, COLS> {
    pub fn from_array(contents: [[T; COLS]; ROWS], ctx: &Arc<C>) -> Self {
        let contents = contents
            .into_iter()
            .map(|v| v.into_iter())
            .flatten()
            .collect();

        Gen2DArray {
            contents,
            context: Arc::clone(ctx),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T, C, const ROWS: usize, const COLS: usize> Gen2DArray<T, C, ROWS, COLS>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    pub fn hadamard_prod(mut self, rhs: &Gen2DArray<T, C, ROWS, COLS>) -> Self {
        self.contents
            .iter_mut()
            .zip(rhs.contents.iter())
            .for_each(|(v1, v2)| *v1 = (v1 as &T) * v2);

        self
    }
}

impl<T: Clone, C, const ROWS: usize, const COLS: usize> Gen2DArray<T, C, ROWS, COLS> {
    pub fn transpose(&self) -> Gen2DArray<T, C, COLS, ROWS> {
        let contents = (0..COLS)
            .map(|col| (0..ROWS).map(move |row| self.contents[row * COLS + col].clone()))
            .flatten()
            .collect();

        Gen2DArray {
            contents,
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

/*
impl<
        'a,
        T: 'static + DefaultWithContext<C> + Clone,
        C,
        const N1: usize,
        const N2: usize,
        const N3: usize,
    > Mul<&'a Gen2DArray<T, C, N2, N3>> for &'a Gen2DArray<T, C, N1, N2>
where
    T: AddAssign<T>,
    for<'b> &'b T: Add<&'b T, Output = T>,
    for<'b> &'b T: Sub<&'b T, Output = T>,
    for<'b> &'b T: Mul<&'b T, Output = T>,
{
    type Output = Gen2DArray<T, C, N1, N3>;
    fn mul(self, rhs: &'a Gen2DArray<T, C, N2, N3>) -> Self::Output {
        let n1 = (N1 as f32).log2();
        let n2 = (N2 as f32).log2();
        let n3 = (N3 as f32).log2();

        let m1_diff = (n1 - n2).abs();
        let m2_diff = (n2 - n3).abs();
        let mx_diff = (n1 - n3).abs();

        if m1_diff > 2.0 || m2_diff > 2.0 || mx_diff > 2.0 {
            let def: T = DefaultWithContext::default_ctx(&self.context);
            let mut new_arr: [[T; N3]; N1] = array![_ => array![_ =>  def.clone(); N3]; N1];

            for i in 0..N1 {
                for j in 0..N3 {
                    for k in 0..N2 {
                        new_arr[i][j] += &self.contents[i][k] * &rhs.contents[k][j];
                    }
                }
            }

            return Self::Output {
                contents: new_arr,
                context: Arc::clone(&self.context),
            };
        }

        let v = T::default_ctx(&self.context);
        let a: Vec<Vec<Option<&T>>> = self
            .contents
            .iter()
            .map(|l1| l1.iter().map(|v| Some(v)).collect())
            .collect();
        let b: Vec<Vec<Option<&T>>> = rhs
            .contents
            .iter()
            .map(|l1| l1.iter().map(|v| Some(v)).collect())
            .collect();

        let res = strassen::strassen_matrix_multiplication::<T, &T, &T, C, _, _, _>(
            0,
            &self.context,
            a.as_slice(),
            b.as_slice(),
            (N1, N2),
            (N2, N3),
            &v,
            |v| v.map(|v| *v),
            |v| v.map(|v| *v),
            |v| v,
            |v| v.map(|v| *v),
        );

        let new_arr: [[T; N3]; N1] =
            array![i => array![j =>  res[i][j].as_ref().unwrap_or(&v).clone(); N3]; N1];

        Self::Output {
            contents: new_arr,
            context: Arc::clone(&self.context),
        }
    }
}
*/

impl<T: BooleanType<C>, C, const S: usize, const P: usize, const N1: usize, const N2: usize>
    Gen2DArray<FixedPointNumber<S, P, T, C>, C, N1, N2>
{
    pub fn mul_opt<const N3: usize>(
        mut self,
        rhs: &Gen2DArray<FixedPointNumber<S, P, T, C>, C, N2, N3>,
    ) -> Gen2DArray<FixedPointNumber<S, P, T, C>, C, N1, N3> {
        let self_abs: Vec<_> = self
            .contents
            .into_iter()
            .map(|v| (v.msb().clone()))
            .collect();
        self.contents.iter_mut().for_each(|v| v.abs_assign());

        let rhs_abs: Vec<_> = rhs
            .contents
            .into_iter()
            .map(|v| (v.msb(), v.clone().abs()))
            .collect();

        let mut lst: Vec<FixedPointNumber<S, P, T, C>> = Vec::with_capacity(N1 * N3);

        let false_val = T::from_ctx(false, &self.context);

        for i in 0..N1 {
            for j in 0..N3 {
                let mut val = None;

                for k in 0..N2 {
                    let self_msb = &self_abs[i * N2 + k];
                    let a = &self.contents[i * N2 + k];
                    let (rhs_msb, b) = &rhs_abs[k * N3 + j];
                    let res: FixedPointNumber<S, P, T, C> = a * (self_msb, b, *rhs_msb, &false_val);

                    match val.as_mut() {
                        Some(v1) => {
                            *v1 += &res;
                        }
                        None => {
                            val = Some(res);
                        }
                    }
                }
                lst.push(val.unwrap());
            }
        }

        Gen2DArray {
            contents: lst.try_into().expect("Should convert Vec<_> to Box<[_]>"),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
    pub fn mul_opt_rhs<const N3: usize>(
        &self,
        mut rhs: Gen2DArray<FixedPointNumber<S, P, T, C>, C, N2, N3>,
    ) -> Gen2DArray<FixedPointNumber<S, P, T, C>, C, N1, N3> {
        let rhs_abs: Vec<_> = rhs
            .contents
            .into_iter()
            .map(|v| (v.msb().clone()))
            .collect();
        rhs.contents.iter_mut().for_each(|v| v.abs_assign());

        let lhs_abs: Vec<_> = self
            .contents
            .into_iter()
            .map(|v| (v.msb(), v.clone().abs()))
            .collect();

        let mut lst: Vec<Option<FixedPointNumber<S, P, T, C>>> =
            (0..(N1 * N3)).map(|_| None).collect();

        let false_val = T::from_ctx(false, &self.context);

        for i in 0..N1 {
            for j in 0..N3 {
                let mut val = None;

                for k in 0..N2 {
                    let (self_msb, a) = &lhs_abs[i * N2 + k];
                    let rhs_msb = &rhs_abs[k * N3 + j];
                    let b = &rhs.contents[k * N3 + j];

                    let res: FixedPointNumber<S, P, T, C> = a * (*self_msb, b, rhs_msb, &false_val);

                    match val.as_mut() {
                        Some(v1) => {
                            *v1 += &res;
                        }
                        None => {
                            val = Some(res);
                        }
                    }
                }
                lst[i * N3 + j] = val;
            }
        }

        Gen2DArray {
            contents: lst.into_iter().collect::<Option<_>>().unwrap(),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T, C, const N1: usize, const N2: usize, const N3: usize> Mul<&Gen2DArray<T, C, N2, N3>>
    for &Gen2DArray<T, C, N1, N2>
where
    for<'a> T: AddAssign<&'a T>,
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    type Output = Gen2DArray<T, C, N1, N3>;
    fn mul(self, rhs: &Gen2DArray<T, C, N2, N3>) -> Self::Output {
        let mut lst: Vec<Option<T>> = (0..(N1 * N3)).map(|_| None).collect();

        for i in 0..N1 {
            for j in 0..N3 {
                let mut val = None;

                for k in 0..N2 {
                    let res: T = &self.contents[i * N2 + k] * &rhs.contents[k * N3 + j];
                    match val.as_mut() {
                        Some(v1) => {
                            *v1 += &res;
                        }
                        None => {
                            val = Some(res);
                        }
                    }
                }

                lst[i * N3 + j] = val;
            }
        }

        Self::Output {
            contents: lst.into_iter().collect::<Option<_>>().unwrap(),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<
        T: BooleanType<C>,
        C,
        const BITS: usize,
        const PRECISION: usize,
        const N1: usize,
        const N2: usize,
    > Gen2DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, N1, N2>
{
    pub fn multiply_add_execute<const BITS_INT: usize, const N3: usize, T1>(
        &self,
        multiply: &Gen2DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, N2, N3>,
        add: &Gen2DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, N1, N3>,
        execute: impl Fn(Gen2DArray<FixedPointNumber<BITS_INT, PRECISION, T, C>, C, N1, N3>) -> T1,
    ) -> T1 {
        let mut self_abs = Vec::with_capacity(self.contents.len());
        let mut rhs_abs = Vec::with_capacity(multiply.contents.len());

        self.contents
            .into_iter()
            .for_each(|v| self_abs.push((v.msb(), v.clone().abs())));
        multiply
            .contents
            .into_iter()
            .for_each(|v| rhs_abs.push((v.msb(), v.clone().abs())));

        let mut lst: Box<[FixedPointNumber<BITS_INT, PRECISION, T, C>]> =
            add.contents.into_iter().map(|v| v.update_size()).collect();
        let lst_mut = lst.as_mut();

        let false_val = T::from_ctx(false, &self.context);

        for i in 0..N1 {
            for j in 0..N3 {
                let v = lst_mut.get_mut(i * N3 + j).unwrap();
                for k in 0..N2 {
                    let (self_msb, a) = &self_abs[i * N2 + k];
                    let (rhs_msb, b) = &rhs_abs[k * N3 + j];
                    let res: FixedPointNumber<BITS, PRECISION, T, C> =
                        a * (*self_msb, b, *rhs_msb, &false_val);
                    let res: FixedPointNumber<BITS_INT, PRECISION, T, C> = res.update_size();
                    *v += &res;
                }
            }
        }

        execute(Gen2DArray {
            contents: lst,
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        })
    }
}

impl<T: Clone, C, const N1: usize, const N2: usize> Mul<&T> for Gen2DArray<T, C, N1, N2>
where
    for<'a> T: Mul<&'a T, Output = T>,
{
    type Output = Gen2DArray<T, C, N1, N2>;
    fn mul(mut self, rhs: &T) -> Self::Output {
        self.contents
            .iter_mut()
            .for_each(|v1| *v1 = v1.clone() * rhs);
        self
    }
}

impl<T, C, const N1: usize, const N2: usize> Add<&Gen2DArray<T, C, N1, N2>>
    for Gen2DArray<T, C, N1, N2>
where
    for<'a> T: AddAssign<&'a T>,
{
    type Output = Gen2DArray<T, C, N1, N2>;
    fn add(mut self, rhs: &Gen2DArray<T, C, N1, N2>) -> Self::Output {
        self.contents
            .iter_mut()
            .zip(rhs.contents.iter())
            .for_each(|(v1, v2)| *v1 += v2);
        self
    }
}
impl<T, C, const N1: usize, const N2: usize> AddAssign<&Gen2DArray<T, C, N1, N2>>
    for Gen2DArray<T, C, N1, N2>
where
    for<'a> T: AddAssign<&'a T>,
{
    fn add_assign(&mut self, rhs: &Gen2DArray<T, C, N1, N2>) {
        self.contents
            .as_mut()
            .iter_mut()
            .zip(rhs.contents.iter())
            .for_each(|v| T::add_assign(v.0, v.1));
    }
}

impl<T, C, const N1: usize, const N2: usize> Div<&T> for &Gen2DArray<T, C, N1, N2>
where
    for<'a> &'a T: Div<&'a T, Output = T>,
{
    type Output = Gen2DArray<T, C, N1, N2>;
    fn div(self, rhs: &T) -> Self::Output {
        let contents = self.contents.iter().map(|v1| v1 / rhs).collect();
        Self::Output {
            contents,
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T, C, const N1: usize, const N2: usize> DivAssign<&T> for Gen2DArray<T, C, N1, N2>
where
    for<'a> &'a T: Div<&'a T, Output = T>,
{
    fn div_assign(&mut self, rhs: &T) {
        for i in 0..N1 {
            for j in 0..N2 {
                self.contents[i * N2 + j] = &self.contents[i * N2 + j] / rhs;
            }
        }
    }
}

impl<T, C, const N1: usize, const N2: usize> Sub<&Gen2DArray<T, C, N1, N2>>
    for Gen2DArray<T, C, N1, N2>
where
    for<'a> T: SubAssign<&'a T>,
{
    type Output = Gen2DArray<T, C, N1, N2>;
    fn sub(mut self, rhs: &Gen2DArray<T, C, N1, N2>) -> Self::Output {
        self.contents
            .iter_mut()
            .zip(rhs.contents.iter())
            .for_each(|(v1, v2)| *v1 -= v2);
        self
    }
}

impl<T, C, const N1: usize, const N2: usize> SubAssign<&Gen2DArray<T, C, N1, N2>>
    for Gen2DArray<T, C, N1, N2>
where
    for<'a> T: SubAssign<&'a T>,
{
    fn sub_assign(&mut self, rhs: &Gen2DArray<T, C, N1, N2>) {
        self.contents
            .iter_mut()
            .zip(rhs.contents.iter())
            .for_each(|(v1, v2)| *v1 -= v2);
    }
}

impl<T, C, const N1: usize, const N2: usize> Gen2DArray<T, C, N1, N2> {
    pub fn apply<T1>(&self, targetfn: impl Fn(&T) -> T1) -> Gen2DArray<T1, C, N1, N2> {
        Gen2DArray {
            contents: self.contents.iter().map(|v1| (targetfn)(v1)).collect(),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
    pub fn apply_with_context<T1, C1>(
        &self,
        new_context: &Arc<C1>,
        targetfn: impl Fn(&Arc<C1>, &T) -> T1,
    ) -> Gen2DArray<T1, C1, N1, N2> {
        Gen2DArray {
            contents: self
                .contents
                .iter()
                .map(|v1| (targetfn)(new_context, v1))
                .collect(),
            context: Arc::clone(new_context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
    pub fn apply_mut(&mut self, targetfn: impl Fn(&mut T) -> ()) {
        self.contents.iter_mut().for_each(|v1| {
            (targetfn)(v1);
        });
    }
    pub fn apply_zip<T1>(
        &self,
        other: &Self,
        targetfn: impl Fn((&T, &T)) -> T1,
    ) -> Gen2DArray<T1, C, N1, N2> {
        Gen2DArray {
            contents: self
                .contents
                .iter()
                .zip(other.contents.iter())
                .map(|v1| (targetfn)(v1))
                .collect(),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
    pub fn apply_zip_2<T1>(
        &self,
        other1: &Self,
        other2: &Self,
        targetfn: impl Fn((&T, &T, &T)) -> T1,
    ) -> Gen2DArray<T1, C, N1, N2> {
        Gen2DArray {
            contents: self
                .contents
                .iter()
                .zip(other1.contents.iter())
                .zip(other2.contents.iter())
                .map(|((v, v1), v2)| (targetfn)((v, v1, v2)))
                .collect(),
            context: Arc::clone(&self.context),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T: DefaultWithContext<C> + Clone, C, const N1: usize, const N2: usize> DefaultWithContext<C>
    for Gen2DArray<T, C, N1, N2>
{
    fn default_ctx(ctx: &Arc<C>) -> Self {
        let t = T::default_ctx(ctx);
        Gen2DArray {
            contents: (0..(N1 * N2)).map(|_| t.clone()).collect(),
            context: Arc::clone(ctx),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T, C, const N1: usize, const N2: usize> Gen2DArray<T, C, N1, N2> {
    pub fn apply_two(&self, targetfn: impl Fn(&T) -> (T, T)) -> (Self, Self) {
        let (lst1, lst2): (Vec<_>, Vec<_>) = self.contents.iter().map(|v1| (targetfn)(v1)).unzip();
        (
            Gen2DArray {
                contents: lst1
                    .try_into()
                    .expect("Should convert array to vec (apply_two:1)"),
                context: Arc::clone(&self.context),
                rows: PhantomData::default(),
                cols: PhantomData::default(),
            },
            Gen2DArray {
                contents: lst2
                    .try_into()
                    .expect("Should convert array to vec (apply_two:2)"),
                context: Arc::clone(&self.context),
                rows: PhantomData::default(),
                cols: PhantomData::default(),
            },
        )
    }
}

impl<T, C, const N1: usize, const N2: usize> PartialEq<Gen2DArray<T, C, N1, N2>>
    for Gen2DArray<T, C, N1, N2>
where
    for<'a> &'a T: PartialEq<&'a T>,
{
    fn eq(&self, other: &Gen2DArray<T, C, N1, N2>) -> bool {
        for i in 0..N1 {
            for j in 0..N2 {
                if &self.contents[i * N2 + j] != &other.contents[i * N2 + j] {
                    return false;
                }
            }
        }
        return true;
    }
}

impl<T, C, T1, C1, const N1: usize, const N2: usize> FromWithContext<Gen2DArray<T, C, N1, N2>, C1>
    for Gen2DArray<T1, C1, N1, N2>
where
    T: Clone,
    T1: FromWithContext<T, C1>,
{
    fn from_ctx(t: Gen2DArray<T, C, N1, N2>, ctx: &Arc<C1>) -> Self {
        Gen2DArray {
            contents: t
                .contents
                .into_iter()
                .map(|v1| T1::from_ctx(v1.clone(), ctx))
                .collect(),
            context: Arc::clone(&ctx),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

#[cfg(not(feature = "tokio"))]
mod multithreading {
    use crate::DefaultWithContext;

    use super::Gen2DArray;
    use std::{iter::Iterator, ops::AddAssign};

    impl<
            'a,
            T: 'a + Clone + Sync + Send + DefaultWithContext<C>,
            C: 'a + Sync + Send,
            const N1: usize,
            const N2: usize,
        > Gen2DArray<T, C, N1, N2>
    where
        for<'b> T: AddAssign<&'b T>,
    {
        pub fn fold_add<I: Iterator<Item = &'a Self>>(lst: I) -> Self {
            let v: Vec<_> = lst.collect();
            let c = v.get(0).unwrap().context.clone();
            v.into_iter()
                .fold(DefaultWithContext::default_ctx(&c), |v1, v2| {
                    v1.clone() + v2
                })
        }
    }
}

#[cfg(feature = "tokio")]
mod multithreading {
    use super::Gen2DArray;
    use std::{iter::Iterator, ops::AddAssign};

    #[derive(Clone)]
    enum RefOrValue<'a, T> {
        Ref(&'a T),
        Val(T),
    }

    impl<
            'a,
            T: 'a + Clone + Sync + Send,
            C: 'a + Sync + Send,
            const N1: usize,
            const N2: usize,
        > Gen2DArray<T, C, N1, N2>
    where
        for<'b> T: AddAssign<&'b T>,
    {
        pub fn fold_add<I: Iterator<Item = &'a Self>>(lst: I) -> Self {
            use tokio::sync::Mutex;

            let mut curr_lst: Vec<RefOrValue<Self>> = lst.map(|v| RefOrValue::Ref(v)).collect();
            while curr_lst.len() > 1 {
                let chunks: Vec<&[RefOrValue<Self>]> = curr_lst.chunks(2).collect();

                let ret_lst = Mutex::new(Vec::with_capacity(chunks.len()));
                tokio_scoped::scope(|scope| {
                    for lst in chunks {
                        scope.spawn(async {
                            if lst.len() == 1 {
                                ret_lst.lock().await.push(Some(lst[0].clone()));
                            } else {
                                let r = match &lst[0] {
                                    RefOrValue::Ref(v) => (*v).clone(),
                                    RefOrValue::Val(v) => v.clone(),
                                } + match &lst[1] {
                                    RefOrValue::Ref(v) => *v,
                                    RefOrValue::Val(v) => v,
                                };
                                ret_lst.lock().await.push(Some(RefOrValue::Val(r)));
                            }
                        });
                    }
                });

                curr_lst = ret_lst
                    .into_inner()
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .expect("Should all be filled");
            }
            match curr_lst.remove(0) {
                RefOrValue::Ref(r) => r.clone(),
                RefOrValue::Val(r) => r,
            }
        }
    }
}

// TODO: Verify utility when fitting multiple samples per fit (maybe beneficial because matrix are more square)
#[allow(dead_code)]
mod strassen {
    use std::{
        marker::PhantomData,
        ops::{Add, Mul, Sub},
        sync::Arc,
    };

    use crate::{DefaultWithContext, Gen2DArray};

    impl<T: DefaultWithContext<C> + Clone + 'static, C, const N1: usize, const N2: usize>
        Gen2DArray<T, C, N1, N2>
    where
        for<'a> &'a T: Mul<&'a T, Output = T>,
        for<'a> T: Add<&'a T, Output = T>,
        for<'a> T: Sub<&'a T, Output = T>,
    {
        pub fn mul_strassen<const N3: usize>(
            self,
            rhs: &Gen2DArray<T, C, N2, N3>,
        ) -> Gen2DArray<T, C, N1, N3> {
            let max_dim = usize::max(N1.max(N2), N3);
            let log = (max_dim as f32).log2();
            let new_size = match log != log.ceil() {
                true => 2.0_f32.powi(log.ceil() as i32) as usize,
                false => max_dim,
            };

            let a_lst: Vec<Option<&T>> = row_major_to_block_partitioned(
                self.contents.as_ref(),
                (N1, N2),
                (new_size, new_size),
            );
            let b_lst = row_major_to_block_partitioned(
                rhs.contents.as_ref(),
                (N2, N3),
                (new_size, new_size),
            );

            let v = T::default_ctx(&self.context);

            let num_elements = new_size * new_size;
            let mut res: Vec<_> = (0..num_elements).map(|_| None).collect();
            let mut stack: Vec<_> = (0..((num_elements / 4) * 12)).map(|_| None).collect();

            strassen_matrix_multiplication::<T, C>(
                0,
                &self.context,
                &a_lst,
                &b_lst,
                (new_size, new_size),
                &v,
                &mut res,
                &mut stack,
            );

            let res_row = block_partitioned_to_row_major(&res, (N1, N3), (new_size, new_size), &v);

            Gen2DArray {
                contents: res_row
                    .try_into()
                    .expect("Should convert array to vec (from_ctx)"),
                context: Arc::clone(&self.context),
                rows: PhantomData::default(),
                cols: PhantomData::default(),
            }
        }
    }

    fn block_partitioned_to_row_major<'a, T: Clone>(
        matrix: &'a [Option<T>],
        source_size: (usize, usize),
        size: (usize, usize),
        default_val: &T,
    ) -> Vec<T> {
        let mut dest: Vec<_> = (0..(source_size.0 * source_size.1))
            .map(|_| default_val.clone())
            .collect();
        to_row_major(
            matrix,
            &mut dest,
            source_size,
            size,
            2,
            size,
            (0, 0),
            0,
            default_val,
        );
        dest
    }

    fn to_row_major<'a, T: Clone>(
        data: &'a [Option<T>],
        dest: &mut [T],
        source_size: (usize, usize),
        original_size: (usize, usize),
        partition_size: usize,
        size: (usize, usize),
        start: (usize, usize),
        start_block: usize,
        default_val: &T,
    ) {
        let start_pos = (partition_size * partition_size) * start_block;
        if size.0 == 2 {
            for row in 0..partition_size {
                for col in 0..partition_size {
                    let real_row = start.0 + row;
                    let real_col = start.1 + col;

                    if real_row >= source_size.0 || real_col >= source_size.1 {
                        // dest[indx] = None;
                    } else {
                        dest[real_row * source_size.1 + real_col] = data
                            [start_pos + (row * partition_size) + col]
                            .as_ref()
                            .map(|v| v.clone())
                            .unwrap_or_else(|| default_val.clone());
                    }
                }
            }
            return;
        }

        let num_local_blocks = (size.0 * size.1) / (partition_size * partition_size);
        let num_blocks_batch = num_local_blocks / 4;

        let new_size = (size.0 / 2, size.1 / 2);

        to_row_major(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0, start.1),
            start_block,
            default_val,
        );
        to_row_major(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0, start.1 + new_size.1),
            start_block + num_blocks_batch,
            default_val,
        );
        to_row_major(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0 + new_size.0, start.1),
            start_block + num_blocks_batch * 2,
            default_val,
        );
        to_row_major(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0 + new_size.0, start.1 + new_size.1),
            start_block + num_blocks_batch * 3,
            default_val,
        );
    }

    fn row_major_to_block_partitioned<'a, T>(
        matrix: &'a [T],
        source_size: (usize, usize),
        size: (usize, usize),
    ) -> Vec<Option<&'a T>> {
        let mut dest: Vec<_> = (0..(size.0 * size.1)).map(|_| None).collect();
        to_partition(matrix, &mut dest, source_size, size, 2, size, (0, 0), 0);
        dest
    }

    fn to_partition<'a, T>(
        data: &'a [T],
        dest: &mut [Option<&'a T>],
        source_size: (usize, usize),
        original_size: (usize, usize),
        partition_size: usize,
        size: (usize, usize),
        start: (usize, usize),
        start_block: usize,
    ) {
        let start_pos = (partition_size * partition_size) * start_block;
        if size.0 == 2 {
            for row in 0..partition_size {
                for col in 0..partition_size {
                    let real_row = start.0 + row;
                    let real_col = start.1 + col;

                    let indx = start_pos + (row * partition_size) + col;

                    if real_row >= source_size.0 || real_col >= source_size.1 {
                        dest[indx] = None;
                    } else {
                        dest[start_pos + (row * partition_size) + col] =
                            Some(&data[real_row * source_size.1 + real_col]);
                    }
                }
            }
            return;
        }

        let num_local_blocks = (size.0 * size.1) / (partition_size * partition_size);
        let num_blocks_batch = num_local_blocks / 4;

        let new_size = (size.0 / 2, size.1 / 2);

        to_partition(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0, start.1),
            start_block,
        );
        to_partition(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0, start.1 + new_size.1),
            start_block + num_blocks_batch,
        );
        to_partition(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0 + new_size.0, start.1),
            start_block + num_blocks_batch * 2,
        );
        to_partition(
            data,
            dest,
            source_size,
            original_size,
            partition_size,
            new_size,
            (start.0 + new_size.0, start.1 + new_size.1),
            start_block + num_blocks_batch * 3,
        );
    }

    pub fn strassen_matrix_multiplication<'a, T: 'static + Clone, C>(
        iteration: usize,
        ctx: &Arc<C>,
        original_a: &[Option<&'a T>],
        original_b: &[Option<&'a T>],
        size: (usize, usize),
        zero_val: &T,
        dest: &mut [Option<T>],
        stack: &mut [Option<T>],
    ) where
        for<'b> T: Add<&'b T, Output = T>,
        for<'b> T: Sub<&'b T, Output = T>,
        for<'b> &'b T: Mul<&'b T, Output = T>,
    {
        if size.0 == 1 {
            dest[0] = match (original_a[0], original_b[0]) {
                (Some(a), Some(b)) => Some(a * b),
                _ => None,
            };
            return;
        }

        let num_elements = size.0 * size.1;
        let chunk_size = num_elements / 4;
        let new_size = (size.0 / 2, size.1 / 2);

        let mut a_chunks = original_a.chunks(chunk_size);
        let mut b_chunks = original_b.chunks(chunk_size);
        let mut dest_chunks = dest.chunks_mut(chunk_size);

        let a11 = a_chunks.next().unwrap();
        let a12 = a_chunks.next().unwrap();
        let a21 = a_chunks.next().unwrap();
        let a22 = a_chunks.next().unwrap();

        let b11 = b_chunks.next().unwrap();
        let b12 = b_chunks.next().unwrap();
        let b21 = b_chunks.next().unwrap();
        let b22 = b_chunks.next().unwrap();

        let c11 = dest_chunks.next().unwrap();
        let c12 = dest_chunks.next().unwrap();
        let c21 = dest_chunks.next().unwrap();
        let c22 = dest_chunks.next().unwrap();

        let (tmp_1, stack) = stack.split_at_mut(chunk_size);
        let (tmp_2, stack) = stack.split_at_mut(chunk_size);

        let (m1, stack) = stack.split_at_mut(chunk_size);
        let (m2, stack) = stack.split_at_mut(chunk_size);
        let (m3, stack) = stack.split_at_mut(chunk_size);
        let (m4, stack) = stack.split_at_mut(chunk_size);
        let (m5, stack) = stack.split_at_mut(chunk_size);
        let (m6, stack) = stack.split_at_mut(chunk_size);
        let (m7, stack) = stack.split_at_mut(chunk_size);

        let mut has_m3 = true;
        let mut has_m5 = true;
        let mut has_m6 = true;
        let mut has_m7 = true;

        let start = std::time::Instant::now();
        if matrix_opt(a11, a22, zero_val, Operation::Sum, tmp_1)
            && matrix_opt(b11, b22, zero_val, Operation::Sum, tmp_2)
        {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                &tmp_1.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                &tmp_2.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                new_size,
                zero_val,
                m1,
                stack,
            );
        } else {
            m1.iter_mut().for_each(|v| *v = None);
        }
        if iteration == 0 {
            println!("DUR {}", start.elapsed().as_secs_f64());
        }

        if validate_has_values(&b11) && matrix_opt(a21, a22, zero_val, Operation::Sum, tmp_1) {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                &tmp_1.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                b11,
                new_size,
                zero_val,
                m2,
                stack,
            );
        } else {
            m2.iter_mut().for_each(|v| *v = None);
        }

        if validate_has_values(&a11) && matrix_opt(b12, b22, zero_val, Operation::Sub, tmp_2) {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                a11,
                &tmp_2.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                new_size,
                zero_val,
                m3,
                stack,
            );
        } else {
            has_m3 = false;
            m3.iter_mut().for_each(|v| *v = None);
        }

        if validate_has_values(&a22) && matrix_opt(b21, b11, zero_val, Operation::Sub, tmp_2) {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                a22,
                &tmp_2.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                new_size,
                zero_val,
                m4,
                stack,
            );
        } else {
            m4.iter_mut().for_each(|v| *v = None);
        }

        if validate_has_values(&b22) && matrix_opt(a11, a12, zero_val, Operation::Sum, tmp_1) {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                &tmp_1.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                b22,
                new_size,
                zero_val,
                m5,
                stack,
            );
        } else {
            has_m5 = false;
            m5.iter_mut().for_each(|v| *v = None);
        }

        if matrix_opt(a21, a11, zero_val, Operation::Sub, tmp_1)
            && matrix_opt(b11, b12, zero_val, Operation::Sum, tmp_2)
        {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                &tmp_1.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                &tmp_2.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                new_size,
                zero_val,
                m6,
                stack,
            );
        } else {
            has_m6 = false;
        }

        if matrix_opt(a12, a22, zero_val, Operation::Sub, tmp_1)
            && matrix_opt(b21, b22, zero_val, Operation::Sum, tmp_2)
        {
            strassen_matrix_multiplication::<T, C>(
                iteration + 1,
                ctx,
                &tmp_1.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                &tmp_2.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                new_size,
                zero_val,
                m7,
                stack,
            );
        } else {
            has_m7 = false;
        }

        // C11
        matrix_opt_dest(m1, m4, zero_val, Operation::Sum, c11);
        if has_m5 {
            matrix_op_dest_acc(m5, zero_val, Operation::Sub, c11);
        }
        if has_m7 {
            matrix_op_dest_acc(m7, zero_val, Operation::Sum, c11);
        }

        // C12
        matrix_opt_dest(m3, m5, zero_val, Operation::Sum, c12);

        // C21
        matrix_opt_dest(m2, m4, zero_val, Operation::Sum, c21);

        // C22
        matrix_opt_dest(m1, m2, zero_val, Operation::Sub, c22);
        if has_m3 {
            matrix_op_dest_acc(m3, zero_val, Operation::Sum, c22);
        }
        if has_m6 {
            matrix_op_dest_acc(m6, zero_val, Operation::Sum, c22);
        }
    }

    #[inline(always)]
    fn validate_has_values<T: Clone>(a: &[Option<&T>]) -> bool {
        for v in a {
            if v.is_some() {
                return true;
            }
        }
        return false;
    }

    #[inline(always)]
    fn matrix_opt<T: Clone>(
        a: &[Option<&T>],
        b: &[Option<&T>],
        zero_val: &T,
        opr: Operation,
        output: &mut [Option<T>],
    ) -> bool
    where
        for<'b> T: Add<&'b T, Output = T>,
        for<'b> T: Sub<&'b T, Output = T>,
    {
        let mut has_vals = false;
        output
            .iter_mut()
            .zip(a.iter().zip(b.iter()))
            .for_each(|(dest, (a_val, b_val))| {
                if a_val.is_some() || b_val.is_some() {
                    has_vals = true;
                }
                *dest = match b_val.as_ref() {
                    Some(b_val) => match a_val.as_ref() {
                        Some(a_val) => Some(match opr {
                            Operation::Sum => (*a_val).clone() + *b_val,
                            Operation::Sub => (*a_val).clone() - *b_val,
                        }),
                        None => Some(match opr {
                            Operation::Sum => (*b_val).clone(),
                            Operation::Sub => zero_val.clone() - *b_val,
                        }),
                    },
                    None => match a_val.as_ref() {
                        Some(a_val) => Some((*a_val).clone()),
                        None => None,
                    },
                }
            });
        has_vals
    }

    #[inline(always)]
    fn matrix_opt_dest<T: Clone>(
        a: &[Option<T>],
        b: &[Option<T>],
        zero_val: &T,
        opr: Operation,
        dest: &mut [Option<T>],
    ) where
        for<'b> T: Add<&'b T, Output = T>,
        for<'b> T: Sub<&'b T, Output = T>,
    {
        dest.iter_mut()
            .zip(a.iter().zip(b.iter()))
            .for_each(|(dest, (a_val, b_val))| {
                *dest = match b_val.as_ref() {
                    Some(b_val) => match a_val.as_ref() {
                        Some(a_val) => Some(match opr {
                            Operation::Sum => a_val.clone() + b_val,
                            Operation::Sub => a_val.clone() - b_val,
                        }),
                        None => Some(match opr {
                            Operation::Sum => (*b_val).clone(),
                            Operation::Sub => zero_val.clone() - b_val,
                        }),
                    },
                    None => match a_val.as_ref() {
                        Some(a_val) => Some((*a_val).clone()),
                        None => None,
                    },
                }
            });
    }

    #[inline(always)]
    fn matrix_op_dest_acc<T: Clone>(
        a: &[Option<T>],
        zero_val: &T,
        opr: Operation,
        dest: &mut [Option<T>],
    ) where
        for<'b> T: Add<&'b T, Output = T>,
        for<'b> T: Sub<&'b T, Output = T>,
    {
        dest.iter_mut().zip(a.iter()).for_each(|(dest, b_val)| {
            *dest = match b_val.as_ref() {
                Some(b_val) => match dest.as_ref() {
                    Some(a_val) => Some(match opr {
                        Operation::Sum => a_val.clone() + b_val,
                        Operation::Sub => a_val.clone() - b_val,
                    }),
                    None => Some(match opr {
                        Operation::Sum => (*b_val).clone(),
                        Operation::Sub => zero_val.clone() - b_val,
                    }),
                },
                None => match dest.as_ref() {
                    Some(a_val) => Some((*a_val).clone()),
                    None => None,
                },
            }
        });
    }

    enum Operation {
        Sum,
        Sub,
    }
}

#[cfg(test)]
mod test {
    use std::{sync::Arc, time::Instant};

    use crate::{DefaultWithContext, Gen2DArray};

    impl DefaultWithContext<()> for i32 {
        fn default_ctx(_ctx: &Arc<()>) -> Self {
            0
        }
    }

    #[test]
    fn test_mat_transpose() {
        let ctx = Arc::new(());

        assert_eq!(
            Gen2DArray::from_array([[0, 1, 2, 3], [4, 5, 6, 7]], &ctx).transpose(),
            Gen2DArray::from_array([[0, 4], [1, 5], [2, 6], [3, 7]], &ctx)
        )
    }

    #[test]
    fn test_mat_harm_prod() {
        let ctx = Arc::new(());

        assert_eq!(
            Gen2DArray::from_array([[0, 2, 2, 4], [4, 6, 6, 8]], &ctx)
                .hadamard_prod(&Gen2DArray::from_array([[0, 2, 2, 4], [4, 6, 6, 8]], &ctx)),
            Gen2DArray::from_array([[0, 4, 4, 16], [16, 36, 36, 64]], &ctx)
        )
    }

    #[test]
    fn test_mat_div() {
        let ctx = Arc::new(());

        assert_eq!(
            &Gen2DArray::from_array([[0, 2, 2, 4], [4, 6, 6, 8]], &ctx) / &2,
            Gen2DArray::from_array([[0, 1, 1, 2], [2, 3, 3, 4]], &ctx)
        )
    }

    #[test]
    fn test_mat_add() {
        let ctx = Arc::new(());

        assert_eq!(
            Gen2DArray::from_array([[2, 6, 12], [45, 5, 2], [5, 5, 2]], &ctx)
                + &Gen2DArray::from_array([[0, 1, 2], [5, 3, 10], [6, 0, 4]], &ctx),
            Gen2DArray::from_array([[2, 7, 14], [50, 8, 12], [11, 5, 6]], &ctx)
        )
    }

    #[test]
    fn test_mat_mult() {
        let ctx = Arc::new(());

        assert_eq!(
            &Gen2DArray::from_array([[1, 2], [3, 4]], &ctx)
                * &Gen2DArray::from_array([[5, 6], [7, 8]], &ctx),
            Gen2DArray::from_array([[19, 22], [43, 50]], &ctx),
            "2x2 * 2x2 matrix multiplication"
        );
        assert_eq!(
            &Gen2DArray::from_array([[2, 6, 12], [45, 5, 2], [5, 5, 2]], &ctx)
                * &Gen2DArray::from_array([[0, 1], [5, 3], [6, 0]], &ctx),
            Gen2DArray::from_array([[102, 20], [37, 60], [37, 20]], &ctx)
        );
        assert_eq!(
            &Gen2DArray::from_array([[2, 6, 12], [10, 5, 2], [5, 2, 3]], &ctx)
                * &Gen2DArray::from_array([[1], [5], [6]], &ctx),
            Gen2DArray::from_array([[104], [47], [33]], &ctx)
        );
    }

    #[test]
    fn test_mat_mult_small() {
        let ctx = Arc::new(());

        let arr1 = Gen2DArray::from_array([[1, 2], [3, 4]], &ctx);

        let arr2 = Gen2DArray::from_array([[1, 2], [3, 4]], &ctx);

        let exp = Gen2DArray::from_array([[7, 10], [15, 22]], &ctx);

        let start = Instant::now();
        let received = arr1.mul_strassen(&arr2);
        // let received = &arr1 * &arr2;
        println!(
            "Duration {}s",
            Instant::now().duration_since(start).as_secs_f64()
        );

        assert_eq!(received, exp);
    }

    #[test]
    fn test_mat_mult_big() {
        let ctx = Arc::new(());

        let arr1 = Gen2DArray::from_array(
            [
                [12, 7, 10, 4, 5, 8, 14, 16, 9, 7, 8, 6, 14, 17, 6],
                [6, 11, 7, 10, 9, 5, 8, 14, 12, 6, 7, 10, 8, 14, 27],
                [10, 9, 8, 7, 6, 11, 5, 8, 14, 10, 9, 7, 14, 12, 4],
                [11, 5, 7, 8, 6, 10, 9, 7, 6, 11, 8, 14, 10, 9, 0],
                [7, 6, 11, 9, 8, 7, 6, 10, 5, 8, 14, 10, 9, 7, 3],
                [9, 7, 8, 14, 11, 6, 7, 9, 8, 7, 6, 11, 10, 8, 9],
                [8, 14, 10, 9, 7, 8, 6, 7, 6, 10, 5, 8, 14, 10, 5],
                [5, 8, 14, 10, 9, 7, 6, 11, 9, 8, 7, 6, 10, 9, 3],
                [6, 7, 9, 8, 7, 5, 8, 7, 6, 11, 9, 8, 7, 6, 2],
                [8, 7, 6, 10, 5, 8, 14, 10, 9, 7, 6, 10, 5, 8, 12],
                [10, 9, 8, 7, 6, 11, 9, 8, 7, 6, 11, 9, 8, 7, 24],
                [7, 6, 10, 9, 8, 7, 6, 11, 9, 8, 7, 6, 10, 9, 8],
                [5, 8, 7, 6, 11, 9, 8, 7, 6, 10, 9, 8, 7, 6, 15],
                [9, 8, 7, 6, 5, 10, 9, 8, 7, 6, 11, 9, 8, 7, 9],
                [8, 7, 6, 11, 9, 8, 7, 6, 5, 10, 9, 8, 7, 6, 9],
            ],
            &ctx,
        );

        let arr2 = Gen2DArray::from_array(
            [
                [
                    1, 5, 7, 9, 5, 6, 0, 1, 9, 3, 5, 4, 9, 8, 8, 1, 3, 4, 3, 4, 7, 9, 4, 1, 3,
                ],
                [
                    7, 2, 6, 2, 2, 9, 7, 3, 2, 4, 2, 1, 2, 0, 3, 6, 4, 7, 4, 3, 1, 8, 0, 6, 6,
                ],
                [
                    4, 5, 4, 5, 2, 3, 1, 3, 7, 9, 8, 8, 3, 2, 0, 1, 4, 8, 8, 5, 4, 9, 3, 0, 4,
                ],
                [
                    4, 2, 9, 4, 9, 7, 7, 3, 9, 8, 5, 4, 0, 9, 2, 1, 6, 3, 6, 1, 7, 7, 0, 1, 2,
                ],
                [
                    4, 7, 2, 7, 4, 0, 3, 4, 5, 0, 0, 9, 8, 8, 2, 6, 0, 5, 1, 2, 8, 0, 3, 0, 7,
                ],
                [
                    3, 5, 8, 0, 0, 7, 0, 4, 7, 8, 0, 3, 3, 0, 5, 3, 4, 9, 2, 1, 2, 3, 9, 8, 1,
                ],
                [
                    6, 8, 2, 4, 4, 5, 7, 4, 9, 8, 0, 1, 8, 2, 6, 1, 4, 3, 1, 0, 2, 6, 2, 3, 2,
                ],
                [
                    7, 9, 8, 0, 8, 5, 4, 7, 6, 6, 3, 9, 0, 2, 8, 8, 5, 5, 7, 0, 5, 5, 4, 0, 6,
                ],
                [
                    6, 1, 2, 1, 0, 6, 7, 4, 1, 4, 2, 8, 7, 9, 5, 8, 2, 1, 2, 2, 8, 4, 6, 1, 3,
                ],
                [
                    7, 3, 5, 5, 8, 4, 7, 9, 4, 2, 1, 6, 5, 4, 5, 3, 7, 8, 7, 5, 2, 3, 8, 9, 9,
                ],
                [
                    8, 3, 2, 7, 5, 6, 3, 4, 3, 7, 0, 2, 4, 8, 5, 1, 1, 3, 7, 0, 1, 1, 6, 8, 6,
                ],
                [
                    0, 0, 4, 1, 8, 5, 4, 2, 0, 8, 5, 0, 1, 8, 9, 2, 7, 7, 7, 3, 0, 2, 3, 5, 9,
                ],
                [
                    2, 6, 2, 0, 8, 9, 4, 8, 7, 6, 5, 7, 1, 3, 1, 6, 5, 5, 3, 5, 2, 7, 4, 0, 2,
                ],
                [
                    8, 8, 1, 6, 1, 0, 4, 6, 8, 2, 3, 5, 3, 3, 9, 5, 2, 4, 0, 0, 4, 8, 0, 8, 2,
                ],
                [
                    9, 9, 4, 3, 8, 0, 3, 7, 2, 8, 8, 0, 2, 0, 5, 8, 3, 4, 4, 7, 2, 2, 2, 0, 8,
                ],
            ],
            &ctx,
        );

        let exp = Gen2DArray::from_array(
            [
                [
                    742, 791, 594, 505, 655, 685, 566, 694, 838, 778, 446, 691, 541, 566, 758, 594,
                    535, 701, 546, 324, 525, 793, 501, 466, 595,
                ],
                [
                    889, 839, 658, 512, 793, 631, 649, 763, 730, 874, 578, 633, 504, 602, 788, 753,
                    564, 726, 620, 438, 560, 723, 467, 430, 782,
                ],
                [
                    662, 618, 573, 456, 592, 695, 544, 634, 710, 705, 402, 653, 502, 605, 652, 558,
                    506, 676, 529, 334, 505, 697, 526, 481, 574,
                ],
                [
                    532, 538, 541, 439, 596, 626, 482, 537, 669, 666, 352, 538, 460, 585, 633, 409,
                    503, 643, 514, 283, 421, 612, 473, 467, 548,
                ],
                [
                    587, 557, 529, 450, 603, 598, 468, 542, 640, 686, 368, 561, 418, 572, 571, 429,
                    463, 622, 558, 279, 424, 583, 450, 421, 571,
                ],
                [
                    623, 623, 590, 467, 679, 618, 539, 587, 697, 721, 444, 596, 459, 624, 618, 521,
                    509, 643, 546, 336, 513, 645, 428, 372, 610,
                ],
                [
                    618, 596, 569, 433, 608, 660, 532, 597, 687, 683, 416, 584, 432, 517, 576, 512,
                    513, 684, 527, 346, 443, 695, 431, 439, 575,
                ],
                [
                    612, 589, 540, 425, 567, 596, 502, 569, 673, 674, 393, 636, 424, 538, 541, 490,
                    470, 633, 532, 296, 479, 644, 432, 376, 544,
                ],
                [
                    528, 480, 463, 400, 525, 529, 455, 488, 565, 580, 309, 495, 400, 497, 501, 380,
                    424, 552, 477, 257, 379, 528, 395, 385, 509,
                ],
                [
                    652, 631, 565, 427, 622, 588, 536, 570, 663, 736, 402, 501, 465, 530, 656, 499,
                    494, 606, 507, 303, 449, 616, 425, 400, 579,
                ],
                [
                    760, 741, 622, 491, 709, 625, 525, 660, 687, 845, 509, 514, 494, 533, 698, 599,
                    516, 699, 581, 416, 459, 645, 494, 436, 693,
                ],
                [
                    621, 611, 535, 423, 592, 571, 489, 581, 655, 668, 402, 597, 428, 529, 572, 509,
                    461, 607, 511, 310, 470, 613, 435, 365, 553,
                ],
                [
                    656, 625, 523, 437, 612, 536, 490, 592, 596, 681, 383, 510, 452, 495, 580, 518,
                    453, 633, 507, 342, 415, 530, 451, 411, 627,
                ],
                [
                    602, 580, 526, 419, 567, 593, 462, 537, 618, 693, 369, 485, 439, 506, 603, 461,
                    450, 603, 497, 295, 400, 579, 445, 420, 550,
                ],
                [
                    584, 552, 531, 440, 600, 554, 475, 535, 614, 643, 360, 494, 428, 533, 553, 439,
                    451, 593, 496, 302, 421, 545, 421, 401, 562,
                ],
            ],
            &ctx,
        );

        let start = Instant::now();
        let received = arr1.mul_strassen(&arr2);
        // let received = &arr1 * &arr2;
        println!(
            "Duration {}s",
            Instant::now().duration_since(start).as_secs_f64()
        );

        assert_eq!(received, exp);
    }
}
