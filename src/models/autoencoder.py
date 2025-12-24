import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()

        # Encoder: compress normal behavior
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Decoder: reconstruct input
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class AutoencoderModel:
    def __init__(self, input_dim: int, lr: float = 1e-3):
        self.model = Autoencoder(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

    def train(self, X_train: np.ndarray, epochs: int = 20, batch_size: int = 64):
        """
        Train autoencoder ONLY on normal data.
        """
        self.model.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32)

        for epoch in range(epochs):
            permutation = torch.randperm(X_tensor.size(0))

            epoch_loss = 0.0
            for i in range(0, X_tensor.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch = X_tensor[indices]

                self.optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error as anomaly score.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

        # Normalize scores to [0, 1]
        scores = errors.numpy()
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores
