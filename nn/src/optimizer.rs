use crate::{matrix::Matrix, parameter::Parameter};
use std::iter::zip;

pub trait Optimizer {
    fn step(&mut self, parameters: Vec<&mut Parameter>);
    
    fn zero_grad(&self, parameters: Vec<&mut Parameter>) {
        for param in parameters {
            param.zero_grad();
        }
    }

    fn get_lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

// Stochastic gradient descent
pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: Vec<&mut Parameter>) {
        for param in parameters {
            param.data = &param.data - &(self.lr * &param.grad);
        }
    }

    fn get_lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}

// Adam optimizer
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    moms1: Vec<Matrix>,
    moms2: Vec<Matrix>,
}

impl Adam {
    pub fn new(parameters: Vec<&mut Parameter>, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        let moms: Vec<Matrix> = parameters
            .iter()
            .map(|p| Matrix::full_like(&p.data, 0.0))
            .collect();
        Self { 
            lr, beta1, beta2, eps, weight_decay,
            t: 0,
            moms1: moms.clone(),
            moms2: moms,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: Vec<&mut Parameter>) {
        self.t += 1;
        for (param, (m1, m2)) in zip(parameters, zip(&mut self.moms1,&mut self.moms2)) {
            // weight decay
            let grad = &(&param.grad + &(self.weight_decay * &param.data));

            // update both moments
            *m1 = &(&*m1 * self.beta1) + &((1.0 - self.beta1) * grad);
            *m2 = &(&*m2 * self.beta2) + &((1.0 - self.beta2) * &(grad * grad));
            
            // correct bias
            let m1_ = &*m1 / (1.0 - self.beta1.powi(self.t as i32));
            let m2_ = &*m2 / (1.0 - self.beta2.powi(self.t as i32));

            // update parameter
            param.data = &param.data - &(self.lr * &(&m1_ / &(&m2_.sqrt() + self.eps)));
        }
    }

    fn get_lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}