# Notebook guide

The notebooks are intentionally small entry points into the importable package. Run them from the repository root after installing the relevant extra. `01_api_controls.ipynb` has a dry-run cell so it can be opened without credentials; the local notebooks require the optional local extra and a model download. `03_test_time_compute.ipynb` and `04_mcts_search.ipynb` run fully offline: they use inline deterministic fakes and tiny arithmetic state spaces.
