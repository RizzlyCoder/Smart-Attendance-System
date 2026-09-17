import pytest

from src.emotion_model import EmotionClassifier
from tests.conftest import make_pattern_image


def test_emotion_classifier_trains_and_predicts(tmp_config, make_dataset_dir):
    dataset_dir = tmp_config / "emotion_dataset"
    dataset_dir.mkdir()
    spec = {
        "happy": [1, 2, 3, 4],
        "sad": [10, 11, 12, 13],
        "neutral": [20, 21, 22, 23],
    }
    make_dataset_dir(str(dataset_dir), spec)

    clf = EmotionClassifier()
    n_images, labels = clf.train_from_directory(str(dataset_dir))

    assert n_images == 12
    assert set(labels) == {"happy", "sad", "neutral"}

    predicted_label, confidence = clf.predict(make_pattern_image(1))
    assert predicted_label in {"happy", "sad", "neutral"}
    assert 0.0 <= confidence <= 1.0


def test_emotion_classifier_requires_two_classes(tmp_config, make_dataset_dir):
    dataset_dir = tmp_config / "emotion_dataset_single"
    dataset_dir.mkdir()
    make_dataset_dir(str(dataset_dir), {"happy": [1, 2, 3]})

    clf = EmotionClassifier()
    with pytest.raises(ValueError):
        clf.train_from_directory(str(dataset_dir))


def test_emotion_classifier_persists_and_reloads(tmp_config, make_dataset_dir):
    dataset_dir = tmp_config / "emotion_dataset"
    dataset_dir.mkdir()
    make_dataset_dir(str(dataset_dir), {"happy": [1, 2, 3, 4], "sad": [10, 11, 12, 13]})

    clf = EmotionClassifier()
    clf.train_from_directory(str(dataset_dir))

    reloaded = EmotionClassifier()
    reloaded.load()
    label, confidence = reloaded.predict(make_pattern_image(1))
    assert label in {"happy", "sad"}
