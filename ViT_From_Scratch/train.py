import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from components import Patch_embeddings, Encoder, VisionTransformer, Data_Ingestion

class BrainTumorTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.config = config
        self.model = VisionTransformer(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            in_channels=config['channels'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_dim=config['mlp_dim'],
            drop_rate=config['drop_rate']
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        # Load Data
        data_loader = Data_Ingestion()
        self.train_loader, self.test_loader = data_loader.get_loaders(config['batch_size'])

        # For storing results
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_loss_list = []
        self.test_loss_list = []

    def train_one_epoch(self):
        self.model.train()
        total_loss, correct = 0, 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (pred.argmax(1) == y).sum().item()

        total_loss /= len(self.train_loader.dataset)
        accuracy = correct / len(self.train_loader.dataset)
        return total_loss, accuracy

    def evaluate(self):
        self.model.eval()
        total_loss, correct = 0, 0

        with torch.inference_mode():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                total_loss += loss.item() * x.size(0)
                correct += (pred.argmax(1) == y).sum().item()

        total_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return total_loss, accuracy

    def plot_results(self):
        plt.figure()
        plt.plot(self.train_acc_list, label="Train Accuracy")
        plt.plot(self.test_acc_list, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Test Accuracy")
        plt.show()

        plt.figure()
        plt.plot(self.train_loss_list, label="Train Loss")
        plt.plot(self.test_loss_list, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Test Loss")
        plt.show()

    def save_model(self, path="vit_brain_tumor.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def run(self):
        print("Starting Training...")
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.evaluate()

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.test_loss_list.append(test_loss)
            self.test_acc_list.append(test_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.4f}")

        self.plot_results()
        self.save_model()


if __name__ == "__main__":
    config = {
        "batch_size": 16,
        "epochs": 50,
        "learning_rate": 1e-4,
        "patch_size": 8,
        "num_classes": 4,
        "image_size": 64,
        "channels": 3,
        "embed_dim": 256,
        "num_heads": 16,
        "depth": 12,
        "mlp_dim": 512,
        "drop_rate": 0.1
    }

    trainer = BrainTumorTrainer(config)
    trainer.run()
