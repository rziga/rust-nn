use crate::optimizer::{self, Optimizer};

pub trait Scheduler {
    fn step(&mut self, optimizer: &mut impl Optimizer);
}

pub struct ExponentialDecay {
    k: f32,
    t: usize,
}

impl ExponentialDecay {
    pub fn new(k: f32) -> Self {
        Self { k, t: 0 }
    }
}

impl Scheduler for ExponentialDecay {
    fn step(&mut self, optimizer: &mut impl Optimizer) {
        self.t += 1;
        let new_lr = optimizer.get_lr() * (-self.k * (self.t as f32)).exp();
        optimizer.set_lr(new_lr);
    }
}