use tch::{Tensor, Kind, nn};
use tch::nn::{Module, Linear, Optimizer, OptimizerConfig};
use crate::algo::vision_transformer::transformer_block::TransformerBlock;

// Vision Transformer Definition
pub struct VisionTransformer {
    patch_embedding: nn::Linear,
    blocks: Vec<TransformerBlock>,
    head: nn::Linear,
    patch_size: i64,
    optimizer: Optimizer,
}

impl VisionTransformer {
    pub fn new(
        vs: &nn::VarStore,
        depth: i64,
        embed_dim: i64,
        num_heads: i64,
        mlp_dim: i64,
        num_classes: i64,
        dropout: f64,
        patch_size: i64,
        learning_rate: f64,
    ) -> Self {
        let root_path = vs.root();
        let patch_dim = 3 * patch_size * patch_size;
        let patch_embedding = nn::linear(&root_path / "patch_embedding", patch_dim, embed_dim, Default::default());
        let head = nn::linear(&root_path / "head", embed_dim, num_classes, Default::default());

        let mut blocks = Vec::new();
        for i in 0..depth {
            blocks.push(TransformerBlock::new(&(&root_path / format!("block_{}", i)), embed_dim, num_heads, mlp_dim, dropout));
        }

        let optimizer = nn::Adam::default().build(vs, learning_rate).expect("Failed to build optimizer");

        VisionTransformer {
            patch_embedding,
            blocks,
            head,
            patch_size,
            optimizer,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        // Correcting view to simulate patch embedding more appropriately
        let x = x.view([-1, 3 * self.patch_size * self.patch_size]); // Flatten patches
        let x = self.patch_embedding.forward(&x);

        // Reshape to [batch size, num patches, embedding dim]
        let num_patches = (224 / self.patch_size) * (224 / self.patch_size);
        let x = x.view([-1, num_patches, 768]);
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x);
        }

        let x = x.mean_dim(&[1_i64][..], false, tch::Kind::Float);
        let output = self.head.forward(&x);

        output
    }

    pub fn train_step(&mut self, inputs: &Tensor, targets: &Tensor) -> f64 {
        let outputs = self.forward(inputs);
        let targets = targets.argmax(-1, false);
        let loss = outputs.cross_entropy_for_logits(&targets);
        loss.backward();
        self.optimizer.step();
        loss.double_value(&[])
    }
}