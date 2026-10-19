"""Regression checks for class-specific maps and unchanged model predictions."""

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sign_mlp.evaluate import gradcam_heatmap, plot_gradcam_by_class, select_class_examples


class GradCamTests(unittest.TestCase):
    def setUp(self):
        # One positive filter; opposite logits explain different classes.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((4, 4, 1)),
            tf.keras.layers.Conv2D(1, 1, activation="relu", name="conv_3"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        self.model.layers[0].set_weights([
            np.ones((1, 1, 1, 1), np.float32), np.zeros(1, np.float32)
        ])
        self.model.layers[-1].set_weights([
            np.array([[100, -100]], np.float32), np.zeros(2, np.float32)
        ])
        self.image = np.arange(16, dtype=np.float32).reshape(4, 4, 1) / 15

    def tearDown(self):
        plt.close("all")
        tf.keras.backend.clear_session()

    def test_saturated_softmax_still_has_spatial_logit_map(self):
        before = self.model(self.image[None], training=False).numpy()
        heatmap, index = gradcam_heatmap(self.model, self.image)
        self.assertEqual(index, 0)
        np.testing.assert_allclose(heatmap, self.image.squeeze(), atol=1e-6)
        np.testing.assert_array_equal(
            before, self.model(self.image[None], training=False).numpy()
        )
        self.assertIs(self.model.layers[-1].activation, tf.keras.activations.softmax)

    def test_chosen_class_and_zero_positive_evidence_are_finite(self):
        heatmap, index = gradcam_heatmap(self.model, self.image, class_index=1)
        self.assertEqual(index, 1)
        self.assertTrue(np.isfinite(heatmap).all())
        np.testing.assert_array_equal(heatmap, np.zeros((4, 4)))

    def test_saved_model_preserves_connected_class_gradients(self):
        expected, _ = gradcam_heatmap(self.model, self.image, class_index=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.keras"
            self.model.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            actual, _ = gradcam_heatmap(restored, self.image, class_index=0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_one_example_for_every_class_even_without_correct_predictions(self):
        truth = np.array([0, 0, 1])
        predictions = np.array([1, 0, 0])
        mapping = {0: 0, 1: 1}
        self.assertEqual(select_class_examples(truth, predictions, mapping), [1, 2])
        fig = plot_gradcam_by_class(
            self.model, np.repeat(self.image[None], 3, axis=0),
            truth, predictions, mapping,
        )
        titles = [ax.get_title() for ax in fig.axes]
        self.assertIn("Grad-CAM clase A", titles)
        self.assertIn("Grad-CAM clase B", titles)
        with self.assertRaisesRegex(ValueError, "No examples"):
            select_class_examples(truth[:1], predictions[:1], mapping)


if __name__ == "__main__":
    unittest.main()
