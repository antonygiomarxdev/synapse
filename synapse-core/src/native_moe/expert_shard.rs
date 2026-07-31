/// Per-expert GGUF loader for distributed MoE inference.
///
/// Loads only specified expert indices from a GGUF file without
/// loading all 40 experts. Each expert's weights are independently
/// addressable by byte offset in the quantized tensor data.
use std::collections::HashMap;
use std::io;
use std::path::Path;

use super::gguf::{GgufFile, GgmlType};
use super::quant::dequantize_expert;

/// Weights for a single expert (gate, up, down projections).
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// A shard of experts loaded from a GGUF file.
///
/// Contains only the specified expert indices for a single layer.
/// Used by expert worker nodes that hold a subset of the full
/// expert pool.
pub struct ExpertShard {
    /// Global expert ID → weights for that expert.
    pub experts: HashMap<usize, ExpertWeights>,
    /// The expert indices loaded in this shard.
    pub indices: Vec<usize>,
    /// Model dimensions (needed for FFN computation).
    pub d_model: usize,
    pub d_ff: usize,
}

impl ExpertShard {
    /// Load only the specified experts from a GGUF file for one layer.
    ///
    /// `expert_indices` are global expert IDs (0..40).
    /// `d_model` and `d_ff` come from the model config.
    pub fn load(
        path: &Path,
        layer: usize,
        expert_indices: &[usize],
        d_model: usize,
        d_ff: usize,
    ) -> io::Result<Self> {
        let gguf = GgufFile::open(path)?;
        let expert_elems = d_model * d_ff;

        let gate_name = format!("blk.{layer}.ffn_gate_exps.weight");
        let up_name = format!("blk.{layer}.ffn_up_exps.weight");
        let down_name = format!("blk.{layer}.ffn_down_exps.weight");

        let gate_info = gguf
            .find_tensor(&gate_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("tensor not found: {gate_name}"),
                )
            })?
            .clone();
        let up_info = gguf
            .find_tensor(&up_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("tensor not found: {up_name}"),
                )
            })?
            .clone();
        let down_info = gguf
            .find_tensor(&down_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("tensor not found: {down_name}"),
                )
            })?
            .clone();

        let gate_offset = gguf.data_file_offset() + gate_info.offset;
        let up_offset = gguf.data_file_offset() + up_info.offset;
        let down_offset = gguf.data_file_offset() + down_info.offset;

        let mut experts = HashMap::new();

        for &eid in expert_indices {
            let gate = dequantize_expert(
                path,
                gate_offset,
                gate_info.ggml_type,
                eid,
                expert_elems,
            )?;
            let up = dequantize_expert(
                path,
                up_offset,
                up_info.ggml_type,
                eid,
                expert_elems,
            )?;
            let down = dequantize_expert(
                path,
                down_offset,
                down_info.ggml_type,
                eid,
                expert_elems,
            )?;

            experts.insert(
                eid,
                ExpertWeights { gate, up, down },
            );
        }

        Ok(ExpertShard {
            experts,
            indices: expert_indices.to_vec(),
            d_model,
            d_ff,
        })
    }

    /// Run expert FFN for the given expert IDs and scores.
    ///
    /// Same math as `expert_ffn` in forward.rs but uses local shard
    /// weights with global-to-local index mapping.
    ///
    /// IMPORTANT: scores must be pre-normalized by the caller.
    /// This function does NOT normalize scores — it trusts the caller
    /// to provide already-normalized weights (sum ≈ 1.0).
    pub fn expert_ffn(
        &self,
        hidden: &[f32],
        expert_ids: &[u32],
        expert_scores: &[f32],
    ) -> Vec<f32> {
        let d_model = self.d_model;
        let d_ff = self.d_ff;

        // Use scores as-is — caller is responsible for normalization
        let norm_scores = expert_scores;

        let mut output = vec![0.0f32; d_model];

        for (ki, &eid) in expert_ids.iter().enumerate() {
            let weights = match self.experts.get(&(eid as usize)) {
                Some(w) => w,
                None => continue, // expert not in this shard
            };
            let score = norm_scores[ki];

            // gate_proj: hidden @ gate^T → [d_ff]
            let mut gate_out = vec![0.0f32; d_ff];
            for j in 0..d_ff {
                let mut acc = 0.0f32;
                for d in 0..d_model {
                    acc += hidden[d]
                        * weights.gate[j * d_model + d];
                }
                gate_out[j] = acc;
            }

            // up_proj: hidden @ up^T → [d_ff]
            let mut up_out = vec![0.0f32; d_ff];
            for j in 0..d_ff {
                let mut acc = 0.0f32;
                for d in 0..d_model {
                    acc += hidden[d]
                        * weights.up[j * d_model + d];
                }
                up_out[j] = acc;
            }

            // SiLU(gate) * up → [d_ff]
            let mut fused = vec![0.0f32; d_ff];
            for j in 0..d_ff {
                fused[j] = silu(gate_out[j]) * up_out[j];
            }

            // down_proj: fused @ down^T → [d_model]
            for d in 0..d_model {
                let mut acc = 0.0f32;
                for j in 0..d_ff {
                    acc += fused[j] * weights.down[d * d_ff + j];
                }
                output[d] += score * acc;
            }
        }

        output
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
        )
    }

    #[test]
    fn load_single_expert() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: model not found");
            return;
        }
        let shard =
            ExpertShard::load(&path, 0, &[0], 1536, 512)
                .unwrap();
        assert_eq!(shard.experts.len(), 1);
        assert!(shard.experts.contains_key(&0));
        let w = &shard.experts[&0];
        assert_eq!(w.gate.len(), 1536 * 512);
        assert_eq!(w.up.len(), 1536 * 512);
        assert_eq!(w.down.len(), 512 * 1536);
        assert!(w.gate.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn load_multiple_experts() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: model not found");
            return;
        }
        let shard = ExpertShard::load(
            &path,
            0,
            &[0, 5, 19, 39],
            1536,
            512,
        )
        .unwrap();
        assert_eq!(shard.experts.len(), 4);
        for eid in [0, 5, 19, 39] {
            assert!(shard.experts.contains_key(&eid));
        }
    }

    #[test]
    fn shard_ffn_produces_output() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: model not found");
            return;
        }
        let shard =
            ExpertShard::load(&path, 0, &[0, 1], 1536, 512)
                .unwrap();
        let hidden = vec![1.0f32; 1536];
        let expert_ids = vec![0u32, 1u32];
        let expert_scores = vec![0.6f32, 0.4f32];
        let out = shard.expert_ffn(&hidden, &expert_ids, &expert_scores);
        assert_eq!(out.len(), 1536);
        assert!(out.iter().any(|&v| v.abs() > 0.001));
    }

    #[test]
    fn missing_expert_returns_zero() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: model not found");
            return;
        }
        // Load only expert 0, but request expert 5
        let shard =
            ExpertShard::load(&path, 0, &[0], 1536, 512)
                .unwrap();
        let hidden = vec![1.0f32; 1536];
        let out = shard.expert_ffn(&hidden, &[5], &[1.0]);
        // Should return zeros since expert 5 is not loaded
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }
}
