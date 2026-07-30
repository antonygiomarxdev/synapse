/// MoE model structure that supports external expert/weight injection.
///
/// The architecture is split into two parts:
///
/// 1. **Fixed (always loaded):** `gate_inp` routing weights (F32, ~240 KB/layer).
///    The coordinator needs these for routing decisions.
///
/// 2. **Injectable (via trait):** attention weights, expert FFN weights,
///    embedding table, output projection. Any backend (GGUF, safetensors,
///    remote coordinator) can provide these at runtime.
///
/// This design follows ADR-0011: the coordinator controls routing externally,
/// and any weight provider can plug in without modifying the forward loop.
use std::path::Path;

use crate::native_moe::gguf::{GgufFile, TensorInfo};
use crate::native_moe::quant::dequantize_tensor;

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

/// A loaded tensor (f32 buffer with shape metadata).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<u64>,
}

impl Tensor {
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[f32] { &self.data }
}

/// Trait for providing expert/attention weights to the forward loop.
///
/// Implementations:
/// - `GgufWeightProvider` — loads from GGUF files (Phase 5)
/// - Future: remote worker, safetensors, custom runtime
pub trait WeightProvider {
    fn provide_attn_norm(&self, layer: usize) -> Option<Tensor>;
    fn provide_attn_q(&self, layer: usize) -> Option<Tensor>;
    fn provide_attn_k(&self, layer: usize) -> Option<Tensor>;
    fn provide_attn_v(&self, layer: usize) -> Option<Tensor>;
    fn provide_attn_output(&self, layer: usize) -> Option<Tensor>;
    fn provide_ffn_norm(&self, layer: usize) -> Option<Tensor>;
    fn provide_gate_inp(&self, layer: usize) -> Option<Tensor>;
    fn provide_gate_exps(&self, layer: usize) -> Option<Tensor>;
    fn provide_up_exps(&self, layer: usize) -> Option<Tensor>;
    fn provide_down_exps(&self, layer: usize) -> Option<Tensor>;
    fn provide_token_embd(&self) -> Option<Tensor>;
    fn provide_output_norm(&self) -> Option<Tensor>;
    fn provide_output(&self) -> Option<Tensor>;
}

/// One transformer layer with MoE expert weights.
///
/// All heavy tensors are Option<Tensor> — the forward loop handles
/// both "loaded" and "not yet loaded / provided externally" gracefully.
#[derive(Debug, Clone)]
pub struct MoeLayer {
    pub index: usize,
    pub attn_norm: Option<Tensor>,
    pub attn_q: Option<Tensor>,
    pub attn_k: Option<Tensor>,
    pub attn_v: Option<Tensor>,
    pub attn_output: Option<Tensor>,
    pub ffn_norm: Option<Tensor>,
    /// Always loaded (F32, ~240 KB). The coordinator needs this.
    pub gate_inp: Tensor,
    pub gate_exps: Option<Tensor>,
    pub up_exps: Option<Tensor>,
    pub down_exps: Option<Tensor>,
}

/// Complete MoE model with optional weight injection.
pub struct MoeModel {
    pub config: MoeConfig,
    pub token_embd: Option<Tensor>,
    pub output_norm: Option<Tensor>,
    pub output: Option<Tensor>,
    pub layers: Vec<MoeLayer>,
}

impl MoeModel {
    /// Load only the routing-critical tensors (gate_inp, norms) from GGUF.
    /// Heavy quantized tensors are deferred — they can be injected later
    /// or loaded on demand.
    pub fn load_routing(path: &Path) -> Result<Self, String> {
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

        // Load lightweight routing-only tensors (F32)
        let try_f32 = |name: &str| -> Option<Tensor> {
            let info = gguf.find_tensor(name)?;
            let abs = gguf.data_file_offset() + info.offset;
            dequantize_tensor(path, abs, info.ggml_type, &info.shape).ok().map(|data| Tensor {
                name: name.to_string(), data, shape: info.shape.clone(),
            })
        };

        let output_norm = try_f32("output_norm.weight");
        let token_embd = try_f32("token_embd.weight");
        let output = try_f32("output.weight");

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for idx in 0..config.n_layers as usize {
            let try_load = |suffix: &str| -> Option<Tensor> {
                let name = format!("blk.{idx}.{suffix}.weight");
                try_f32(&name)
            };

            let gate_inp = try_load("ffn_gate_inp")
                .ok_or(format!("layer {idx}: gate_inp missing"))?;

            layers.push(MoeLayer {
                index: idx,
                attn_norm: try_load("attn_norm"),
                attn_q: None,
                attn_k: None,
                attn_v: None,
                attn_output: None,
                ffn_norm: try_load("ffn_norm"),
                gate_inp,
                gate_exps: None,
                up_exps: None,
                down_exps: None,
            });
        }

        Ok(MoeModel { config, token_embd, output_norm, output, layers })
    }
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
    fn load_routing_loads_config_and_gate_inp() {
        let model = MoeModel::load_routing(&model_path()).expect("load failed");
        let c = &model.config;
        assert_eq!(c.d_model, 1536);
        assert_eq!(c.n_layers, 32);
        assert_eq!(c.n_experts, 40);

        assert_eq!(model.layers.len(), 32);
        let l0 = &model.layers[0];
        assert!(!l0.gate_inp.is_empty());
        assert_eq!(l0.gate_inp.data.len(), 1536 * 40);
    }

    #[test]
    fn load_routing_loads_norms() {
        let model = MoeModel::load_routing(&model_path()).expect("load failed");
        let l0 = &model.layers[0];
        assert!(l0.ffn_norm.is_some());
        assert!(l0.attn_norm.is_some());
    }

    #[test]
    fn weight_provider_trait_is_object_safe() {
        // Compile-time check that the trait supports dyn dispatch
        fn _assert(_: &dyn WeightProvider) {}
    }
}
