import argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.aligner import PrototypeAligner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    data = torch.load("data/processed/embedding.pt")
    dataset = TensorDataset(data["x"], data["pseudo"], data["mask"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    model = PrototypeAligner(data["x"].shape[1], cfg["latent_dim"], data["num_types"])
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    for epoch in range(cfg["epochs"]):
        for x, pseudo, mask in loader:
            optimiser.zero_grad()
            _, logits = model(x)
            loss = model.loss(logits, pseudo, mask.bool())
            loss.backward()
            optimiser.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
