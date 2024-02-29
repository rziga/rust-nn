use crate::matrix::Matrix;

pub struct Parameter {
    pub data: Matrix,
    pub grad: Matrix,
}

impl Parameter {
    pub fn new(data: Matrix) -> Self {
        Self {
            grad: Matrix::full_like(&data, 0.0),
            data,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = &self.grad * 0.0;
    }
}