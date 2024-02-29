use crate::matrix::Matrix;

pub trait Loss {
    fn forward(&mut self, input: &Matrix, target: &Matrix) -> f32;
    fn backward(&self, seed: f32) -> Matrix;
}

// Crossentropy from Logits - LogSoftmax for stability
// assumes that targets are one-hot encoded
pub struct Crossentropy {
    target: Option<Matrix>,
    activation: Option<Matrix>,
}

impl Crossentropy {
    pub fn new() -> Self {
        Self { target: None, activation: None }
    }
}

impl Loss for Crossentropy {
    fn forward(&mut self, input: &Matrix, target: &Matrix) -> f32 {
        // log softmax, normalize for stability
        self.target = Some(target.clone());
        let mut x = input - &input.col_max();
        x = &x - &x.exp().col_sum().ln();
        self.activation = Some(x.clone());

        // crossentropy
        (&x * target).sum()
    }

    fn backward(&self, seed: f32) -> Matrix {
        seed * &(self.activation.as_ref().unwrap() - self.target.as_ref().unwrap())
    }
}
