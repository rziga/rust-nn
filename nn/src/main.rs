mod matrix;
mod layer;
mod parameter;
mod loss;
mod optimizer;
mod lr_scheduler;
mod data;
mod metric;

use crate::{data::BatchIter, layer::{Layer, Linear, ReLU, Sequential}, lr_scheduler::{ExponentialDecay, Scheduler}, matrix::Matrix, metric::accuracy};
use data::CIFAR10;
use loss::{Crossentropy, Loss};
use optimizer::{Adam, Optimizer, SGD};
use rand::{random, rngs::StdRng, SeedableRng};
use std::{fs, iter::Iterator, vec};

fn main() {
    /*
    let logits = Matrix::from_vec(2, 3, vec![1., 1., 1., 1., 1., 1.]);
    let targets = Matrix::from_vec(2, 3, vec![1., 0., 0., 0., 1., 0.]);
    let mut loss = Crossentropy::new();

    let l = loss.forward(&logits, &targets);
    println!("{l}");
    println!("{:#?}", loss.backward(1.0));
    */

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
    //sch.step(&mut optim);
    //println!("{}", optim.get_lr());
    
    //let bytes = fs::read("../data/cifar-10-batches-bin/data_batch_1.bin").unwrap();
    //let mut x = vec![];
    //let mut y = vec![];
    //for chunk in bytes.chunks(3073) {
    //    y.push(chunk[0]);
    //    x.append(chunk[1..].to_vec().as_mut());
    //}
    //println!("{:?}", y[..10].to_vec());

    
    let mut rng = StdRng::seed_from_u64(1337);
    let train_dataset = CIFAR10::new(vec![
        "../data/cifar-10-batches-bin/data_batch_1.bin",
        "../data/cifar-10-batches-bin/data_batch_2.bin",
        "../data/cifar-10-batches-bin/data_batch_3.bin",
        "../data/cifar-10-batches-bin/data_batch_4.bin",
    ]);
    let val_dataset = CIFAR10::new(vec![
        "../data/cifar-10-batches-bin/data_batch_5.bin",
    ]);
    let test_dataset = CIFAR10::new(vec![
        "../data/cifar-10-batches-bin/data_batch_5.bin",
    ]);
    
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(3072, 128, true, &mut rng)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(128, 128, true, &mut rng)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(128, 10, true, &mut rng)),
    ]);
    let mut loss_fn = Crossentropy::new();
    //let mut optim = SGD::new(0.1);
    let mut optim = Adam::new(model.parameters(), 0.01, 0.9, 0.999, 1e-5, 1e-4);

    for epoch in 0..10 {

        // train epoch
        let mut train_accs = vec![];
        let mut losses = vec![];
        for (x, y) in BatchIter::new(128, &train_dataset) {
            let logits = model.forward(&x);
            let loss = loss_fn.forward(&logits, &y);
            println!("loss: {:.3}", loss);
            losses.push(loss);
            train_accs.push(accuracy(&logits, &y));
    
            optim.zero_grad(model.parameters());
            model.backward(&loss_fn.backward(1.0));
            optim.step(model.parameters());
        }
        let train_acc = (train_accs.iter().sum::<f32>()) / (train_accs.len() as f32);
        let train_loss = (losses.iter().sum::<f32>()) / (losses.len() as f32);

        // val epoch
        let mut val_accs = vec![];
        for (x, y) in BatchIter::new(128, &val_dataset) {
            let logits = model.forward(&x);
            val_accs.push(accuracy(&logits, &y));
        }
        let val_acc = (val_accs.iter().sum::<f32>()) / (val_accs.len() as f32);

        println!("epoch: {}, loss: {}, train acc: {}, val acc: {}", epoch, train_loss, train_acc, val_acc);
    }
    
}
