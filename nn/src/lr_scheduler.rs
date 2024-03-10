use crate::optimizer::Optimizer;

pub trait Scheduler {
    fn step(&mut self, optimizer: &mut impl Optimizer);
}

pub struct ExponentialDecay {
    k: f32,
    t: usize,
    initial_lr: f32,
}

impl ExponentialDecay {
    pub fn new(k: f32, optimizer: &impl Optimizer) -> Self {
        Self { k, t: 0, initial_lr: optimizer.get_lr() }
    }
}

impl Scheduler for ExponentialDecay {
    fn step(&mut self, optimizer: &mut impl Optimizer) {
        self.t += 1;
        let new_lr = self.initial_lr * (-self.k * (self.t as f32)).exp();
        optimizer.set_lr(new_lr);
    }
}