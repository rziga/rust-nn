mod matrix;
mod layer;
mod parameter;
mod loss;
mod optimizer;
mod lr_scheduler;
mod data;

use crate::{layer::{Linear, ReLU, Sequential, Layer}, lr_scheduler::{Scheduler, ExponentialDecay}, matrix::Matrix, optimizer::{Adam, Optimizer}};
use rand::{rngs::StdRng, SeedableRng};
use std::{fs, vec};

fn main() {
    //let x = Matrix::full(1, 2, 1.0);
    //let y = Matrix::full(3, 2, 2.0);
    //println!("{:#?}", &x + 10f32);
    //println!("{:#?}", x.matmul(&y.T()));
    
    //let mut rng = StdRng::seed_from_u64(1337);
    //let x = Matrix::random(2, 3, &mut rng);
    //let mut  layer = Linear::new(3, 4, false, &mut rng);
    //let y = layer.forward(&x);
    //let g = layer.backward(&y);
    //println!("{y:#?}, {g:#?}");

    //let mut rng = StdRng::seed_from_u64(1337);
    //let x = Matrix::full(2, 3, 1.0);
    //let mut model = Sequential::new(vec![
    //    Box::new(Linear::new(3, 6, true, &mut rng)),
    //    Box::new(ReLU::new()),
    //    Box::new(Linear::new(6, 10, true, &mut rng)),
    //]);
    //let y = model.forward(&x);
    //model.backward(&Matrix::full_like(&y, 1.0));
    //let mut optim = Adam::new(model.parameters(), 1.0, 0.9, 0.999, 1e-5, 0.0);
    //let mut sch = ExponentialDecay::new(10.0);
    //println!("{:#?}", model.parameters()[0].data);
    //optim.step(model.parameters());
    //println!("{:#?}", model.parameters()[0].data);
    //println!("{}", optim.get_lr());
    //optim.set_lr(sch.step(optim.get_lr()));
    //println!("{}", optim.get_lr());
    
    let bytes = fs::read("../data/cifar-10-batches-bin/data_batch_1.bin").unwrap();
    let mut x = vec![];
    let mut y = vec![];
    for chunk in bytes.chunks(3073) {
        y.push(chunk[0]);
        x.append(chunk[1..].to_vec().as_mut());
    }
    println!("{:?}", y[..10].to_vec());
}
