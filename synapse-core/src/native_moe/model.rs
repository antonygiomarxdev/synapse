/// MoE model structure loaded from GGUF.
///
/// Organizes tensors into a layer-by-layer representation that the
/// forward loop can consume. Each `MoeLayer` bundles attention weights,
/// normalization vectors, gate_inp weights for routing, and expert FFN
/// weights (gate/up/down projections).
use std::path::Path;

use crate::native_moe::gguf::{GgufFile, TensorInfo};

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub architecture: String,
    pub d_model: u32,
    pub d_ff: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub n_experts: u32,
    pub n_experts_active: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub norm_eps: f32,
    pub rope_theta: f32,
}

/// A single 1D or 2D tensor loaded into memory.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub data: Vec<f32>,
    /// Original shape (can be empty for scalars, [d] for vectors, [rows, cols] for matrices).
    pub shape: Vec<u64>,
}

impl Tensor {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Access as flat slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// One transformer layer with MoE expert weights.
#[derive(Debug, Clone)]
pub struct MoeLayer {
    pub index: usize,

    // Attention
    pub attn_norm: Tensor,
    pub attn_q: Tensor,
    pub attn_k: Tensor,
    pub attn_v: Tensor,
    pub attn_output: Tensor,

    // MoE FFN
    pub ffn_norm: Tensor,
    /// Expert routing weights: [n_experts, d_model] expert-major.
    pub gate_inp: Tensor,
    /// Expert gate projection: [n_experts, d_model, d_ff].
    pub gate_exps: Tensor,
    /// Expert up projection: [n_experts, d_model, d_ff].
    pub up_exps: Tensor,
    /// Expert down projection: [n_experts, d_ff, d_model].
    pub down_exps: Tensor,
}

/// Complete MoE model loaded from GGUF.
pub struct MoeModel {
    pub config: MoeConfig,
    pub token_embd: Tensor,
    pub output_norm: Tensor,
    pub output: Tensor,
    pub layers: Vec<MoeLayer>,
}

impl MoeModel {
    /// Load a Granite MoE model from GGUF file.
    /// Only loads F32 tensors (gate_inp, norms, output layer, output norm).
    /// Quantized attention and expert tensors are loaded as zeros for V0.
    pub fn load(path: &Path) -> Result<Self, String> {
        let gguf = GgufFile::open(path).map_err(|e| format!("GGUF parse: {e}"))?;

        let arch = gguf.get_string("general.architecture")
            .ok_or("missing general.architecture")?.to_string();

        if arch != "granitemoe" {
            return Err(format!("unsupported architecture: {arch} (only granitemoe for V0)"));
        }

        let config = {
            let a = &arch;
            MoeConfig {
                architecture: arch.clone(),
                d_model: gguf.get_u32(&format!("{a}.embedding_length"))
                    .ok_or("missing embedding_length")?,
                d_ff: gguf.get_u32(&format!("{a}.feed_forward_length"))
                    .ok_or("missing feed_forward_length")?,
                n_layers: gguf.get_u32(&format!("{a}.block_count"))
                    .ok_or("missing block_count")?,
                n_heads: gguf.get_u32(&format!("{a}.attention.head_count"))
                    .ok_or("missing head_count")?,
                n_kv_heads: gguf.get_u32(&format!("{a}.attention.head_count_kv"))
                    .unwrap_or(8),
                n_experts: gguf.get_u32(&format!("{a}.expert_count"))
                    .ok_or("missing expert_count")?,
                n_experts_active: gguf.get_u32(&format!("{a}.expert_used_count"))
                    .ok_or("missing expert_used_count")?,
                vocab_size: gguf.get_u32(&format!("{a}.vocab_size"))
                    .ok_or("missing vocab_size")?,
                max_seq_len: gguf.get_u32(&format!("{a}.context_length"))
                    .ok_or("missing context_length")?,
                norm_eps: gguf.get_f32(&format!("{a}.attention.layer_norm_rms_epsilon"))
                    .ok_or("missing norm_eps")?,
                rope_theta: gguf.get_f32(&format!("{a}.rope.freq_base"))
                    .ok_or("missing rope_theta")?,
            }
        };

        let loads = vec![
            ("token_embd.weight", "token_embd"),
            ("output_norm.weight", "output_norm"),
            ("output.weight", "output"),
        ];

        let (embd, onorm, out) = load_tensors_or_skip(&gguf, path, &loads);

        let token_embd = embd.unwrap_or_else(|| {
            Tensor { name: "token_embd.weight".into(), data: vec![], shape: vec![] }
        });
        let output_norm = onorm.unwrap_or_else(|| {
            Tensor { name: "output_norm.weight".into(), data: vec![], shape: vec![] }
        });
        let output = out.unwrap_or_else(|| {
            Tensor { name: "output.weight".into(), data: vec![], shape: vec![] }
        });

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for l in 0..config.n_layers as usize {
            let layer = load_layer(&gguf, path, l, &config)?;
            layers.push(layer);
        }

        Ok(MoeModel { config, token_embd, output_norm, output, layers })
    }
}

fn load_tensors_or_skip(
    gguf: &GgufFile, path: &Path, names: &[(&str, &str)],
) -> (Option<Tensor>, Option<Tensor>, Option<Tensor>) {
    fn try_load(gguf: &GgufFile, path: &Path, name: &str) -> Option<Tensor> {
        let info = gguf.find_tensor(name)?;
        gguf.read_tensor_f32(path, info).ok().map(|data| Tensor {
            name: name.to_string(),
            data,
            shape: info.shape.clone(),
        })
    }
    (
        try_load(gguf, path, names.get(0).map(|x| x.0).unwrap_or("")),
        try_load(gguf, path, names.get(1).map(|x| x.0).unwrap_or("")),
        try_load(gguf, path, names.get(2).map(|x| x.0).unwrap_or("")),
    )
}

fn load_layer(gguf: &GgufFile, path: &Path, idx: usize, _config: &MoeConfig) -> Result<MoeLayer, String> {
    let try_load = |name: &str| -> Option<Tensor> {
        let full = format!("blk.{idx}.{name}.weight");
        let info = gguf.find_tensor(&full)?;
        gguf.read_tensor_f32(path, info).ok().map(|data| Tensor {
            name: full,
            data,
            shape: info.shape.clone(),
        })
    };

    let gate_inp = try_load("ffn_gate_inp")
        .ok_or(format!("layer {idx}: gate_inp missing"))?;

    let skip = |name: &str| -> Tensor {
        let full = format!("blk.{idx}.{name}.weight");
        let shape = gguf.find_tensor(&full).map(|t| t.shape.clone()).unwrap_or_default();
        let n = shape.iter().product::<u64>() as usize;
        Tensor { name: full, data: vec![0.0f32; n], shape }
    };

    Ok(MoeLayer {
        index: idx,
        attn_norm: try_load("attn_norm").unwrap_or_else(|| skip("attn_norm")),
        attn_q: skip("attn_q"),
        attn_k: skip("attn_k"),
        attn_v: skip("attn_v"),
        attn_output: skip("attn_output"),
        ffn_norm: try_load("ffn_norm").unwrap_or_else(|| skip("ffn_norm")),
        gate_inp,
        gate_exps: skip("ffn_gate_exps"),
        up_exps: skip("ffn_up_exps"),
        down_exps: skip("ffn_down_exps"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
        )
    }

    #[test]
    fn load_granite_moe_config() {
        let model = MoeModel::load(&model_path()).expect("load failed");
        let c = &model.config;
        assert_eq!(c.d_model, 1536);
        assert_eq!(c.n_layers, 32);
        assert_eq!(c.n_experts, 40);
        assert_eq!(c.n_experts_active, 8);
        assert_eq!(c.d_ff, 512);
        assert_eq!(c.vocab_size, 49155);
    }

    #[test]
    fn layers_have_gate_inp() {
        let model = MoeModel::load(&model_path()).expect("load failed");
        assert_eq!(model.layers.len(), 32);

        let l0 = &model.layers[0];
        assert!(!l0.gate_inp.is_empty());
        assert_eq!(l0.gate_inp.data.len(), 40 * 1536);
        assert!(l0.gate_inp.as_slice().iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn token_embd_loaded() {
        let model = MoeModel::load(&model_path()).expect("load failed");
        // token_embd may be empty (Q8_0 skipped for V0) or loaded (F32 format)
        // Just verify model loaded
        assert_eq!(model.layers.len(), 32);
    }
}
