from pathlib import Path
import unittest
from models.checkpoint_manager import CheckpointManager
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import torch.optim as optim
import torch
import random
from tqdm import tqdm
from transformers import AutoModel


# python -m unittest models.checkpoint_manager_test.TestCheckpointManager -v
class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        # Load model
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Create a dummy optimizer and a dummy scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.1
        )

        # Configuration for CheckpointManager
        self.config = {"name": "TestCheckpointManager"}

        # Initialize CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            self.model, self.optimizer, self.scheduler, self.config
        )

    def test_save_load_idempotency(self):
        # Save the checkpoint
        self.checkpoint_manager.save(epoch=1)
        self.checkpoint_manager.save(epoch=2)

        loaded_epoch = self.checkpoint_manager.load()

        # Check if the loaded epoch is correct
        self.assertEqual(loaded_epoch, 2, "Loaded epoch does not match saved epoch.")

    # python -m unittest models.checkpoint_manager_test.TestCheckpointManager.test_get_latest_checkpoint -v
    def test_get_latest_checkpoint(self):
        checkpoint_files = ["epoch_1.pth", "epoch_11.pth", "epoch_2.pth"]
        checkpoint_manager = CheckpointManager(
            None, None, None, {"name": "TestCheckpointManager"}
        )

        latest_checkpoint = checkpoint_manager.get_latest_checkpoint(checkpoint_files)
        expected_checkpoint = Path("./epoch_11.pth")
        latest_checkpoint = Path(latest_checkpoint)
        self.assertEqual(latest_checkpoint.stem, expected_checkpoint.stem)


# python -m unittest models.checkpoint_manager_test.TestCheckpointManagerTraining -v
class TestCheckpointManagerTraining(unittest.TestCase):
    def setUp(self):
        # Set a fixed seed for reproducibility
        torch.manual_seed(0)
        random.seed(0)

        self.config = {
            "name": "TestCheckpointManagerTraining",
            "architecture": {
                "semantic_search_model": {
                    "device": "cpu",
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                }
            },
        }
        self.wrapped_model = WrappedSentenceTransformerModel(self.config)

        # Create a dummy optimizer
        self.optimizer = optim.Adam(
            self.wrapped_model.get_all_trainable_parameters(), lr=0.001
        )

        # (not used in this test)
        self.scheduler = None
        self.scaler = None

        # Initialize CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            self.wrapped_model.model,
            self.optimizer,
            self.scheduler,
            self.config,
            self.scaler,
        )
        self.query = "What color is the fruit that Alice loves?"
        self.documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        self.target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    def test_training_checkpoint(self):
        loss_function = torch.nn.MSELoss()
        initial_losses = []
        reloaded_losses = []

        # Train for 20 steps and drop checkpoint at step 10
        for step in tqdm(range(20)):
            (
                question_embedding,
                document_embeddings,
            ) = self.wrapped_model.get_query_and_document_embeddings(
                self.query, self.documents
            )
            inner_product = (question_embedding @ document_embeddings.T).squeeze()

            loss = loss_function(inner_product.unsqueeze(0), self.target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == 9:
                # Save checkpoint at step 10
                self.checkpoint_manager.save(epoch=10)

            if step >= 9:
                initial_losses.append(loss.item())

        # Wipe clean
        self.optimizer = None
        self.model = None

        # Reload checkpoint at step 10 and retrain
        self.checkpoint_manager.load()
        new_wrapped_model = WrappedSentenceTransformerModel(
            self.config, self.checkpoint_manager.model
        )
        self.optimizer = self.checkpoint_manager.optimizer
        for step in tqdm(range(10, 20)):
            (
                question_embedding,
                document_embeddings,
            ) = new_wrapped_model.get_query_and_document_embeddings(
                self.query, self.documents
            )
            inner_product = (question_embedding @ document_embeddings.T).squeeze()
            loss = loss_function(inner_product.unsqueeze(0), self.target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            reloaded_losses.append(loss.item())

        # Compare the loss values
        for initial_loss, reloaded_loss in zip(initial_losses, reloaded_losses):
            self.assertAlmostEqual(initial_loss, reloaded_loss, places=5)


# python -m unittest models.checkpoint_manager_test.TestCheckpointManagerTrainingAMP -v
class TestCheckpointManagerTrainingAMP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)

        self.config = {
            "name": "TestCheckpointManagerTrainingAMP",
            "architecture": {
                "semantic_search_model": {
                    "device": "cuda:0",
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                }
            },
        }
        self.wrapped_model = WrappedSentenceTransformerModel(self.config)
        self.optimizer = optim.Adam(
            self.wrapped_model.get_all_trainable_parameters(), lr=0.001
        )
        self.scheduler = None

        # Initialize AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize CheckpointManager with AMP scaler
        self.checkpoint_manager = CheckpointManager(
            self.wrapped_model.model,
            self.optimizer,
            self.scheduler,
            self.config,
            self.scaler,
        )
        self.query = "What color is the fruit that Alice loves?"
        self.documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        self.target = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0]], device=self.wrapped_model.device
        )

    def test_training_checkpoint_amp(self):
        loss_function = torch.nn.MSELoss()
        initial_losses = []
        reloaded_losses = []

        # Training loop with AMP
        for step in tqdm(range(20)):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = self.wrapped_model.get_query_and_document_embeddings(
                    self.query, self.documents
                )
                inner_product = (question_embedding @ document_embeddings.T).squeeze()

                loss = loss_function(inner_product.unsqueeze(0), self.target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if step == 9:
                # Save checkpoint with AMP state
                self.checkpoint_manager.save(epoch=10)

            if step >= 9:
                initial_losses.append(loss.item())

        # Wipe clean
        self.optimizer = None
        self.model = None
        self.scaler = None

        # Reload checkpoint
        self.checkpoint_manager.load()
        new_wrapped_model = WrappedSentenceTransformerModel(
            self.config, self.checkpoint_manager.model
        )
        self.optimizer = self.checkpoint_manager.optimizer
        self.scaler = self.checkpoint_manager.scaler

        # Retraining loop with AMP from checkpoint
        for step in tqdm(range(10, 20)):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = new_wrapped_model.get_query_and_document_embeddings(
                    self.query, self.documents
                )
                inner_product = (question_embedding @ document_embeddings.T).squeeze()
                loss = loss_function(inner_product.unsqueeze(0), self.target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            reloaded_losses.append(loss.item())

        # Compare the loss values
        for initial_loss, reloaded_loss in zip(initial_losses, reloaded_losses):
            self.assertAlmostEqual(initial_loss, reloaded_loss, places=5)


if __name__ == "__main__":
    unittest.main()
