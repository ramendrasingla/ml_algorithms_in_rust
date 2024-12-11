use tch::{nn, Tensor};

pub struct MLP {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dropout: f64,
}

impl MLP {
    pub fn new(vs: &nn::Path, embed_dim: i64, mlp_dim: i64, dropout: f64) -> Self {
        let fc1 = nn::linear(vs / "fc1", embed_dim, mlp_dim, Default::default());
        let fc2 = nn::linear(vs / "fc2", mlp_dim, embed_dim, Default::default());
        MLP { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.apply(&self.fc1).relu().dropout(self.dropout, true).apply(&self.fc2)
    }
}
