use tch::{nn, Tensor};
use tch::nn::{Module, Linear};
use crate::algo::ViT::mhsa::MultiHeadSelfAttention;
use crate::algo::ViT::mlp::MLP;

pub struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    mlp: MLP,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl TransformerBlock {
    pub fn new(vs: &nn::Path, embed_dim: i64, num_heads: i64, mlp_dim: i64, dropout: f64) -> Self {
        let attention = MultiHeadSelfAttention::new(&(vs / "attention"), embed_dim, num_heads);
        let mlp = MLP::new(&(vs / "mlp"), embed_dim, mlp_dim, dropout);
        let norm1 = nn::layer_norm(vs / "norm1", vec![embed_dim], Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![embed_dim], Default::default());

        TransformerBlock { attention, mlp, norm1, norm2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        
        let attn_out = self.attention.forward(x);
        let norm1_out = self.norm1.forward(&attn_out);
        let add_residual1 = &norm1_out + x;
        let mlp_out = self.mlp.forward(&add_residual1);
        let norm2_out = self.norm2.forward(&mlp_out);
        let add_residual2 = &norm2_out + &add_residual1;
        add_residual2
    }

    // pub fn backward(
    //     &self,
    //     attn_predictions: &Tensor,
    //     attn_targets: &Tensor,
    //     mlp_predictions: &Tensor,
    //     mlp_targets: &Tensor,
    //     optimizer: &mut nn::Optimizer,
    // ) {
    //     self.attention.backward(attn_predictions, attn_targets, optimizer);
    //     self.mlp.backward(mlp_predictions, mlp_targets, optimizer);
    // }
}