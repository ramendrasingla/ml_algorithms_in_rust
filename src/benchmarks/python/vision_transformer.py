import torch
from torch import nn
import time
import os
import pandas as pd

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)  # [B, C, H, W]
        x = x.flatten(2)  # Flatten the patch dimensions
        x = x.transpose(1, 2)  # [B, N, C] where N is the number of patches
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # Expand cls token to the batch size
        x = torch.cat((cls_tokens, x), dim=1)  # Prepend the class token
        x += self.position_embeddings  # Add position embeddings
        x = self.transformer_encoder(x)  # Pass through the transformer
        x = x[:, 0]  # Take the output of the class token
        return self.mlp_head(x)

def benchmark_vit():
    os.makedirs("./data/benchmarks", exist_ok=True)
    model = VisionTransformer()
    model.eval()
    input_data = torch.rand(256, 3, 224, 224)
    benchmark_file = "./data/benchmarks/vision_transformer.csv"
    new_data = {"Epoch": [], "Python Duration (seconds)": []}
    duration = []
    with torch.no_grad():
        for epoch in range(10):  # Benchmarking for 10 epochs
            start_time = time.time()
            _ = model(input_data)
            epoch_duration = time.time() - start_time
            duration.append(epoch_duration)
            new_data["Epoch"].append(epoch + 1)
            new_data["Python Duration (seconds)"].append(epoch_duration)
            print(f"Epoch {epoch + 1} Duration: {epoch_duration:.2f} seconds")
    total_duration = sum(duration)
    print(f"Total Python Vision Transformer Benchmark Duration: {total_duration:.2f} seconds")
    new_df = pd.DataFrame(new_data)
    if os.path.exists(benchmark_file):
        existing_df = pd.read_csv(benchmark_file)
        merged_df = pd.merge(existing_df, new_df, on="Epoch", how="left")
    else:
        merged_df = new_df
    merged_df.to_csv(benchmark_file, index=False)
    print(f"Benchmark results logged to {benchmark_file}")

if __name__ == "__main__":
    benchmark_vit()