use rand::Rng;

use crate::matrix::Matrix;
use crate::parameter::Parameter;

// Layer trait
pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, partial: &Matrix) -> Matrix;
    fn parameters(&mut self) -> Vec<&mut Parameter>;
}

// Linear layer
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    input: Option<Matrix>,
}

impl Linear {
    pub fn new<R: Rng>(in_chan: usize, out_chan: usize, bias: bool, rng: &mut R) -> Self {
        let bound = 1.0 / (in_chan as f32).sqrt(); // kaiming unif bound for relu
        Self {
            weight: Parameter::new(bound - &((2.0*bound) * &Matrix::random(in_chan, out_chan, rng))),
            bias: if bias {
                Some(Parameter::new(Matrix::full(1, out_chan,0.0)))
            } else { None },
            input: None
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());
        let x = input.matmul(&self.weight.data);
        match &self.bias {
            Some(bias) => &x + &bias.data,
            None => x
        }
    }
    
    fn backward(&mut self, partial: &Matrix) -> Matrix {
        self.weight.grad = &self.weight.grad + &self.input.as_mut().unwrap().T().matmul(partial);
        if !self.bias.is_none() {
            self.bias.as_mut().unwrap().grad = &self.bias.as_ref().unwrap().grad + &partial.row_sum();
        }
        partial.matmul(&self.weight.data.T())
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        match &mut self.bias {
            Some(bias) => vec![&mut self.weight, bias],
            None => vec![&mut self.weight]
        }
    }
}

// ReLU layer
pub struct ReLU {
    mask: Option<Matrix>,
}

impl ReLU {
    pub fn new() -> Self {
        Self{ mask: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.mask = Some(input.maximum(0.0));
        input * self.mask.as_ref().unwrap()
    }

    fn backward(&mut self, partial: &Matrix) -> Matrix {
        partial * self.mask.as_ref().expect("Cannot call backward before forward.")
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        vec![]
    }
}

// Sequential layer - run layers sequentially
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }
}

impl Layer for Sequential {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut x = input.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x);
        }
        x
    }
    
    fn backward(&mut self, partial: &Matrix) -> Matrix {
        let mut x = partial.clone();
        for layer in self.layers.iter_mut().rev() {
            x = layer.backward(&x);
        }
        x
    }

    fn parameters(&mut self) -> Vec<&mut Parameter> {
        let mut params = vec![];
        for layer in self.layers.iter_mut() {
            params.append(&mut layer.parameters());
        }
        params
    }
}