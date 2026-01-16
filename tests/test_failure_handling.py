import pytest
import os
from src.load_data import load_heart_disease

def test_failure_handling(monkeypatch):
    """
    Test to verify pipeline handles dataset loading failure safely.
    """

    # Mock os.path.join to return a wrong file path
    def mock_path(*args, **kwargs):
        return "data/non_existing_file.csv"

    monkeypatch.setattr(os.path, "join", mock_path)

    # Expect an exception when loading fails
    with pytest.raises(Exception):
        load_heart_disease()

