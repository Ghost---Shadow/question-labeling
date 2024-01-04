from pathlib import Path
import shutil
import unittest
from models.checkpoint_manager import CheckpointManager
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import torch.optim as optim
import torch
import random
from tqdm import tqdm


def rmrf_if_possible(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        ...


# python -m unittest models.checkpoint_manager_test.TestCheckpointManagerAutoLoad -v
class TestCheckpointManagerAutoLoad(unittest.TestCase):
    def test_checkpoint_auto_load(self):
        config = {
            "name": "TestCheckpointManagerAutoLoad",
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "name": "sentence_transformer",
                    "device": "cpu",
                }
            },
            "training": {"learning_rate": 1e-4},
        }
        seed = 42

        # Wipe checkpoint dir to test clean load
        checkpoint_dir = CheckpointManager.generate_checkpoint_dir(config, seed)
        rmrf_if_possible(checkpoint_dir)

        checkpoint_manager = CheckpointManager(config, seed)
        checkpoint_manager.save(epoch=2)
        self.assertEqual(checkpoint_manager.last_epoch, 2)

        new_checkpoint_manager = CheckpointManager(config, seed)
        self.assertEqual(new_checkpoint_manager.last_epoch, 2)


# python -m unittest models.checkpoint_manager_test.TestCheckpointManager -v
class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        # Configuration for CheckpointManager
        self.config = {
            "name": "TestCheckpointManagerAutoLoad",
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "name": "sentence_transformer",
                    "device": "cpu",
                }
            },
            "training": {"learning_rate": 1e-4},
        }
        self.seed = 42

        # Initialize CheckpointManager
        self.checkpoint_manager = CheckpointManager(self.config, self.seed)

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
        checkpoint_manager = CheckpointManager(self.config, self.seed)

        latest_checkpoint = checkpoint_manager.get_latest_checkpoint(checkpoint_files)
        expected_checkpoint = Path("./epoch_11.pth")
        latest_checkpoint = Path(latest_checkpoint)
        self.assertEqual(latest_checkpoint.stem, expected_checkpoint.stem)


# python -m unittest models.checkpoint_manager_test.TestCheckpointManagerTraining -v
class TestCheckpointManagerTraining(unittest.TestCase):
    def setUp(self):
        self.config = {
            "name": "TestCheckpointManagerTraining",
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "name": "sentence_transformer",
                    "device": "cuda:0",
                }
            },
            "training": {"learning_rate": 1e-4},
        }
        self.seed = 42
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Wipe checkpoint dir to test clean load
        checkpoint_dir = CheckpointManager.generate_checkpoint_dir(
            self.config, self.seed
        )
        rmrf_if_possible(checkpoint_dir)

        self.checkpoint_manager = CheckpointManager(
            self.config,
            self.seed,
        )
        self.query = "What color is the fruit that Alice loves?"
        self.documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        self.target = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0]], device=self.checkpoint_manager.wrapped_model.device
        )

    def test_training_checkpoint(self):
        loss_function = torch.nn.MSELoss()
        initial_losses = []
        reloaded_losses = []

        # aliases
        wrapped_model = self.checkpoint_manager.wrapped_model
        optimizer = self.checkpoint_manager.optimizer

        # Train for 20 steps and drop checkpoint at step 10
        for step in tqdm(range(20)):
            (
                question_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings(
                self.query, self.documents
            )
            inner_product = (question_embedding @ document_embeddings.T).squeeze()

            loss = loss_function(inner_product.unsqueeze(0), self.target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 9:
                # Save checkpoint at step 10
                self.checkpoint_manager.save(epoch=10)

            if step > 9:
                initial_losses.append(loss.item())

        # Wipe clean
        wrapped_model = None
        optimizer = None

        # Reload checkpoint at step 10 and retrain
        self.checkpoint_manager.load()
        new_wrapped_model = self.checkpoint_manager.wrapped_model
        new_optimizer = self.checkpoint_manager.optimizer

        for step in tqdm(range(10, 20)):
            (
                question_embedding,
                document_embeddings,
            ) = new_wrapped_model.get_query_and_document_embeddings(
                self.query, self.documents
            )
            inner_product = (question_embedding @ document_embeddings.T).squeeze()
            loss = loss_function(inner_product.unsqueeze(0), self.target)
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()

            reloaded_losses.append(loss.item())

        # Compare the loss values
        for initial_loss, reloaded_loss in zip(initial_losses, reloaded_losses):
            self.assertAlmostEqual(initial_loss, reloaded_loss, places=5)


# python -m unittest models.checkpoint_manager_test.TestCheckpointManagerTrainingAMP -v
class TestCheckpointManagerTrainingAMP(unittest.TestCase):
    def setUp(self):
        self.config = {
            "name": "TestCheckpointManagerTrainingAMP",
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "name": "sentence_transformer",
                    "device": "cuda:0",
                }
            },
            "training": {"learning_rate": 1e-4},
        }
        self.seed = 42
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Wipe checkpoint dir to test clean load
        checkpoint_dir = CheckpointManager.generate_checkpoint_dir(
            self.config, self.seed
        )
        rmrf_if_possible(checkpoint_dir)

        # Initialize CheckpointManager with AMP scaler
        self.checkpoint_manager = CheckpointManager(
            self.config,
            self.seed,
        )
        self.query = "What color is the fruit that Alice loves?"
        self.documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        self.target = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0]], device=self.checkpoint_manager.wrapped_model.device
        )

    def test_training_checkpoint_amp(self):
        loss_function = torch.nn.MSELoss()
        initial_losses = []
        reloaded_losses = []

        # aliases
        wrapped_model = self.checkpoint_manager.wrapped_model
        optimizer = self.checkpoint_manager.optimizer
        scaler = self.checkpoint_manager.scaler

        # Training loop with AMP
        for step in tqdm(range(20)):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings(
                    self.query, self.documents
                )
                inner_product = (question_embedding @ document_embeddings.T).squeeze()

                loss = loss_function(inner_product.unsqueeze(0), self.target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step == 9:
                # Save checkpoint with AMP state
                self.checkpoint_manager.save(epoch=10)

            if step > 9:
                initial_losses.append(loss.item())

        # Wipe clean
        optimizer = None
        wrapped_model = None
        scaler = None

        # Reload checkpoint
        self.checkpoint_manager.load()
        new_wrapped_model = self.checkpoint_manager.wrapped_model
        new_optimizer = self.checkpoint_manager.optimizer
        new_scaler = self.checkpoint_manager.scaler

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

            new_optimizer.zero_grad()
            new_scaler.scale(loss).backward()
            new_scaler.step(new_optimizer)
            new_scaler.update()

            reloaded_losses.append(loss.item())

        # Compare the loss values
        # print(initial_losses)
        # print(reloaded_losses)
        for initial_loss, reloaded_loss in zip(initial_losses, reloaded_losses):
            self.assertAlmostEqual(initial_loss, reloaded_loss, places=5)


if __name__ == "__main__":
    unittest.main()
