pub trait Scheduler {
    fn step(&mut self, lr: f32) -> f32;
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
    fn step(&mut self, lr: f32) -> f32 {
        self.t += 1;
        lr * (-self.k * (self.t as f32)).exp()
    }
}