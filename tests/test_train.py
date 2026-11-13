import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tiny_timbre.model import Model
from tiny_timbre.train import (grad_norms, load_checkpoint, predict, save_checkpoint,
                               train_one_epoch)


def _fake_setup():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 128, 64)              # 8 fake short spectrograms
    y = (torch.rand(8, 97) < 0.05).float()      # fake labels, ~5% ones (like real tags)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    model = Model(head=nn.Linear(256, 97))      # stand-in for Ayesha's FlatHead
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return loader, model, optimizer


def test_loss_goes_down_on_fake_data():
    loader, model, optimizer = _fake_setup()
    loss_fn = nn.BCEWithLogitsLoss()            # sigmoid + binary cross-entropy in one

    first = train_one_epoch(model, loader, optimizer, loss_fn)
    for _ in range(49):
        last = train_one_epoch(model, loader, optimizer, loss_fn)
    assert last < first / 2


def test_predict_returns_probabilities():
    loader, model, _ = _fake_setup()
    scores, labels = predict(model, loader)
    assert scores.shape == (8, 97) and labels.shape == (8, 97)
    assert scores.min() >= 0 and scores.max() <= 1      # sigmoid was applied


def test_grad_norms_has_every_layer():
    loader, model, _ = _fake_setup()
    x, y = next(iter(loader))
    nn.BCEWithLogitsLoss()(model(x), y).backward()
    norms = grad_norms(model)
    assert "encoder.blocks.0.0.weight" in norms         # first conv layer
    assert "head.weight" in norms
    assert all(v >= 0 for v in norms.values())


def test_checkpoint_round_trip(tmp_path):
    loader, model, optimizer = _fake_setup()
    train_one_epoch(model, loader, optimizer, nn.BCEWithLogitsLoss())
    save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, epoch=3)

    _, fresh_model, fresh_optimizer = _fake_setup()
    next_epoch = load_checkpoint(tmp_path / "ckpt.pt", fresh_model, fresh_optimizer)
    assert next_epoch == 4
    for a, b in zip(model.state_dict().values(), fresh_model.state_dict().values()):
        assert torch.equal(a, b)


def test_norms_out_collects_one_dict_per_batch():
    loader, model, optimizer = _fake_setup()          # 8 songs, batch size 8 -> 1 batch
    norms = []
    train_one_epoch(model, loader, optimizer, nn.BCEWithLogitsLoss(), norms_out=norms)
    assert len(norms) == 1
    assert "encoder.blocks.0.0.weight" in norms[0]


def test_run_end_to_end_on_fake_data(tmp_path):
    from tiny_timbre.train import run

    loader, _, _ = _fake_setup()
    config = {"name": "test", "head": "flat", "seed": 0, "lr": 1e-3, "epochs": 2,
              "checkpoint_path": str(tmp_path / "ckpt" / "test.pt"),
              "wandb_mode": "disabled"}              # no W&B account needed in tests

    run(config, train_loader=loader, validation_loader=loader)
    assert (tmp_path / "ckpt" / "test.pt").exists()
    run_id = (tmp_path / "ckpt" / "test.pt.wandb_id").read_text()

    run(config, train_loader=loader, validation_loader=loader)   # resumes: nothing left to do
    assert torch.load(tmp_path / "ckpt" / "test.pt")["epoch"] == 1
    assert (tmp_path / "ckpt" / "test.pt.wandb_id").read_text() == run_id   # same W&B run
