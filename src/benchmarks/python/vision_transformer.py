import torch
from torch import nn, optim
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
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=dropout)
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings
        for layer in self.encoder_layers:
            x = layer(x)
        x = x[:, 0]
        return self.mlp_head(x)

    def train_step(self, input_data, target_data, optimizer, criterion):
        optimizer.zero_grad()
        outputs = self.forward(input_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()
        return loss.item()

def benchmark_vit():
    os.makedirs("./data/benchmarks", exist_ok=True)
    model = VisionTransformer()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    input_data = torch.rand(256, 3, 224, 224)
    target_data = torch.randint(0, 10, (256,))
    benchmark_file = "./data/benchmarks/vision_transformer_train.csv"
    new_data = {"Epoch": [], "Python Loss": [], "Python Duration (seconds)": []}

    for epoch in range(10):
        start_time = time.time()
        loss = model.train_step(input_data, target_data, optimizer, criterion)
        epoch_duration = time.time() - start_time
        new_data["Epoch"].append(epoch + 1)
        new_data["Python Loss"].append(loss)
        new_data["Python Duration (seconds)"].append(epoch_duration)
        print(f"Epoch {epoch + 1}: Loss: {loss:.4f}, Duration: {epoch_duration:.2f} seconds")

    new_df = pd.DataFrame(new_data)

    if os.path.exists(benchmark_file):
        existing_df = pd.read_csv(benchmark_file)
        # Merge by 'Epoch', replacing existing loss and duration columns with new data
        existing_df = existing_df.drop(columns=["Python Loss", "Python Duration (seconds)"], errors='ignore')
        merged_df = pd.merge(existing_df, new_df, on="Epoch", how="right")
    else:
        merged_df = new_df

    merged_df.to_csv(benchmark_file, index=False)
    print(f"Training benchmark results logged to {benchmark_file}")

if __name__ == "__main__":
    benchmark_vit()