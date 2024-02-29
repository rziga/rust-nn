use std::ops::{Add, Div, Mul, Neg, Sub};
use std::iter::zip;
use std::vec;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<f32>,
    strides: Vec<usize>,
    pub shape: Vec<usize>,
}

impl Matrix {
    pub fn from_vec(rows: usize, cols: usize, vec: Vec<f32>) -> Self{
        if vec.len() != rows * cols { panic!("Size mismatch.") }
        Self {
            data: vec,
            strides: vec![cols, 1],
            shape: vec![rows, cols],
        }
    }
    
    pub fn full(rows: usize, cols: usize, value: f32) -> Self {
        Self {
            data: vec![value; rows * cols],
            strides: vec![cols, 1],
            shape: vec![rows, cols],
        }
    }

    pub fn full_like(other: &Self, value: f32) -> Self {
        Self::full(other.shape[0], other.shape[1], value)
    }

    pub fn random<R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Matrix {
        Self::from_vec(rows, cols, (0..rows*cols).map(|_| rng.gen::<f32>()).collect())
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn is_row(&self) -> bool {
        self.rows() == 1
    }

    pub fn is_col(&self) -> bool {
        self.cols() == 1
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[self.strides[0] * row + self.strides[1] * col]
    }

    // broadcasted version of get
    fn get_b(&self, row: usize, col: usize) -> f32 {
        let row_ = broadcast_idx(self.rows(), row);
        let col_ = broadcast_idx(self.cols(), col);
        self.get(row_, col_)
    }

    fn get_row(&self, row: usize) -> Self {
        let mut out: Vec<f32> = vec![];
        for i in 0..self.cols() {
            out.push(self.get(row, i));
        }
        Self::from_vec(1, self.cols(), out)
    }

    fn get_col(&self, col: usize) -> Self {
        self.T().get_row(col).T()
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[self.strides[0] * row + self.strides[1] * col] = value;
    }

    // TODO maybe no clone is needed
    pub fn T(&self) -> Self {
        let mut out = Self {
            data: self.data.clone(),
            strides: self.strides.clone(),
            shape: self.shape.clone(),
        };
        out.strides.reverse();
        out.shape.reverse();
        out
    }

    pub fn apply_unary<F: Fn(&f32) -> f32>(&self, fn_: F) -> Self{
        Self {
            data: self.data.iter().map(fn_).collect(),
            strides: self.strides.clone(),
            shape: self.shape.clone(),
        }
    }

    pub fn apply_binary<F: Fn(&f32, &f32) -> f32>(&self, other: &Self, fn_: F) -> Self {
        let r = broadcast_shape(self.rows(), other.rows());
        let c = broadcast_shape(self.cols(), other.cols());
        let mut out = Self::full(r, c, 0.0);
        for i in 0..r {
            for j in 0..c {
                let s = self.get_b(i, j);
                let o = other.get_b(i, j);
                out.set(i, j, fn_(&s, &o));
            }
        }
        out
    }
}


// helpers

fn broadcast_shape(dim1: usize, dim2: usize) -> usize {
    if dim1 == 1 || dim1 == dim2 { return dim2 }
    if dim2 == 1 { return dim1 }
    panic!("Cannot broadcast {dim1} to {dim2}.");
}

fn broadcast_idx(dim: usize, idx: usize) -> usize {
    if dim == 1 { return 0 }
    if idx < dim { return idx } 
    panic!("Cannot broadcast {idx} to {dim}.");
}


// OP overloads

// add
impl Add<f32> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: f32) -> Self::Output {
        self.apply_unary(|x| x + rhs)
    }
}
impl Add<&Matrix> for f32 {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        rhs.apply_unary(|x| self + x)
    }
}
impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        self.apply_binary(rhs, |x, y| x + y)
    }
}

// mul
impl Mul<f32> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f32) -> Self::Output {
        self.apply_unary(|x| x * rhs)
    }
}
impl Mul<&Matrix> for f32 {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs.apply_unary(|x| self * x)
    }
}
impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.apply_binary(rhs, |x, y| x * y)
    }
}

// div
impl Div<f32> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: f32) -> Self::Output {
        self.apply_unary(|x| x / rhs)
    }
}
impl Div<&Matrix> for f32 {
    type Output = Matrix;
    fn div(self, rhs: &Matrix) -> Self::Output {
        rhs.apply_unary(|x| self / x)
    }
}
impl Div<&Matrix> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: &Matrix) -> Self::Output {
        self.apply_binary(rhs, |x, y| x / y)
    }
}

// sub
impl Sub<f32> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: f32) -> Self::Output {
        self.apply_unary(|x| x - rhs)
    }
}
impl Sub<&Matrix> for f32 {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        rhs.apply_unary(|x| self - x)
    }
}
impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        self.apply_binary(rhs, |x, y| x - y)
    }
}

// neg
impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {
        self.apply_unary(|x| -x)
    }
}

// exp and log, maximum
impl Matrix {
    pub fn dot(&self, other: &Self) -> f32 {
        if !(self.is_row() && other.is_col()) { panic!("Shape mismatch in dot.") }
        zip(&self.data, &other.data).map(|(x, y)| x * y).sum()
    }

    pub fn matmul(&self, other: &Self) -> Self {
        if self.cols() != other.rows() { panic!("Shape mismatch in matmul.") }
        let mut out = Matrix::full(self.rows(), other.cols(), 0.0);
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                out.set(i, j, self.get_row(i).dot(&other.get_col(j)));
            }
        }
        out
    }

    // sums
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }
    pub fn row_sum(&self) -> Matrix {
        Self::from_vec(1, self.cols(), 
            (0..self.cols())
            .map(|r| self.get_col(r).sum())
            .collect()
        )
    }
    pub fn col_sum(&self) -> Matrix {
        self.T().row_sum().T()
    }

    // maxes
    pub fn max(&self) -> f32 {
        self.data.iter()
            .reduce(|a, b| if b > a { b } else { a })
            .expect("idk vro something went wrong")
            .clone()
    }
    pub fn row_max(&self) -> Matrix {
        Self::from_vec(1, self.cols(), 
            (0..self.cols())
            .map(|r| self.get_col(r).max())
            .collect()
        )
    }
    pub fn col_max(&self) -> Matrix {
        self.T().row_max().T()
    }

    // exp and log
    pub fn exp(&self) -> Self {
        self.apply_unary(|x| x.exp())
    }
    pub fn ln(&self) -> Self {
        self.apply_unary(|x| x.ln())
    }

    // sqrt
    pub fn sqrt(&self) -> Self {
        self.apply_unary(|x| x.sqrt())
    }

    // maximum
    pub fn maximum(&self, other: f32) -> Self {
        self.apply_unary(|x| (x >= &other) as i32 as f32)
    }
}