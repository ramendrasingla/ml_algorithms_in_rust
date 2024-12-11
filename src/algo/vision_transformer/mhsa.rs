use tch::{nn, Tensor};

pub struct MultiHeadSelfAttention {
    qkv: nn::Linear,
    proj: nn::Linear,
    num_heads: i64,
}

impl MultiHeadSelfAttention {
    pub fn new(vs: &nn::Path, embed_dim: i64, num_heads: i64) -> Self {
        let qkv = nn::linear(vs / "qkv", embed_dim, embed_dim * 3, Default::default());
        let proj = nn::linear(vs / "proj", embed_dim, embed_dim, Default::default());
        MultiHeadSelfAttention { qkv, proj, num_heads }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.size()[0];
        let seq_len = x.size()[1];
        let embed_dim = x.size()[2];
        let head_dim = embed_dim / self.num_heads;

        // Compute Q, K, V
        let qkv = x.apply(&self.qkv).view([batch_size, seq_len, 3, self.num_heads, head_dim]);
        let q = qkv.select(2, 0);
        let k = qkv.select(2, 1);
        let v = qkv.select(2, 2);

        // Attention scores
        let scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        let attn = scores.softmax(-1, scores.kind());

        // Context vector
        attn.matmul(&v).view([batch_size, seq_len, embed_dim]).apply(&self.proj)
    }
}