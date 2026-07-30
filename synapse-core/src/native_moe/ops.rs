/// Low-level tensor operations for the MoE forward pass.
///
/// All operations are pure f32, no dependencies on ggml or external libraries.
/// Used as building blocks for the forward loop.

/// Vector-matrix multiplication: y[j] = sum_d x[d] * w[j * k + d]
/// x: [k], w: [n, k] row-major, returns [n]
pub fn matvec(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0;
        for d in 0..k {
            acc += x[d] * w[j * k + d];
        }
        y[j] = acc;
    }
    y
}

/// RMS Normalization: y = x / sqrt(mean(x²) + eps) * weight
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (ss / d as f32 + eps).sqrt();
    x.iter().zip(weight).map(|(&v, &w)| v * inv_rms * w).collect()
}

/// Softmax
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&v| v / sum).collect()
}

/// Top-K selection: returns indices sorted by descending value.
pub fn top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_zero_input_gives_zero() {
        let x = vec![0.0f32; 4];
        let w = vec![1.0f32; 4];
        let y = rms_norm(&x, &w, 1e-6);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = softmax(&x);
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
