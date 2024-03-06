use crate::matrix::Matrix;

pub fn accuracy(y: &Matrix, target: &Matrix) -> f32 {
    let mut correct = 0.0;
    for i in 0..y.rows() {
        let max_idx = (0..y.cols())
            .map( |j| y.get(i, j) )
            .enumerate()
            .max_by( |(_, a), (_, b)| a.total_cmp(b) )
            .unwrap().0;
        correct += (target.get(i, max_idx) == 1.0f32) as i32 as f32;
    }
    correct / (y.rows() as f32)
}