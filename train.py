"""Smoke-test entry point: build loaders and print one batch's shapes."""

from __future__ import annotations

import numpy as np

from tearcls.data import build_loaders


def main() -> None:
    train_dl, val_dl, test_dl = build_loaders(
        batch_size=4, num_workers=0, collate_fn=lambda b: b
    )
    print(f"train: {len(train_dl.dataset)} samples")
    print(f"val:   {len(val_dl.dataset)} samples")
    print(f"test:  {len(test_dl.dataset)} samples")

    for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        batch = next(iter(dl))
        arr = np.array(batch[0]["image"])
        print(f"{name} batch: size={len(batch)}, img shape={arr.shape}, label={batch[0]['label']}")


if __name__ == "__main__":
    main()
