"""Reproducible Sliding-Window Collaborator Selection for Flower Strategies."""

from collections.abc import Iterable
from logging import INFO
from math import ceil
from random import Random
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
from flwr.supercore import log


def select_sliding_window(
    client_ids: list[int], server_round: int, fraction: float = 0.2, seed: int = 42
) -> list[int]:
    """Select non-overlapping client windows per cycle, reshuffled every cycle."""
    ordered = sorted(int(cid) for cid in client_ids)
    w_size = max(1, int(len(ordered) * fraction))
    n_windows = ceil(len(ordered) / w_size)
    cycle, w_idx = divmod(max(server_round - 1, 0), n_windows)

    # Pad final short window so every round has equal client count
    pad = (-len(ordered)) % w_size
    padded = ordered + ordered[:pad] if pad else ordered
    Random(seed + cycle).shuffle(padded)
    return padded[w_idx * w_size : (w_idx + 1) * w_size]


class SlidingWindowTrainMixin:
    """Strategy mixin that replaces random sampling with cyclic sliding-window selection."""

    def __init__(
        self,
        *args: Any,
        collaborator_selection_fraction: float = 0.2,
        collaborator_selection_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        if not 0.0 < collaborator_selection_fraction <= 1.0:
            raise ValueError("collaborator_selection_fraction must be in (0, 1]")
        self.fraction = collaborator_selection_fraction
        self.seed = collaborator_selection_seed
        self.collaborator_selection_audit: list[dict] = []
        super().__init__(*args, **kwargs)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        node_ids = list(grid.get_node_ids())
        selected = select_sliding_window(node_ids, server_round, self.fraction, self.seed)
        if not selected:
            return []

        # Audit trail for research tracking
        self.collaborator_selection_audit.append(
            {
                "server_round": server_round,
                "selected_node_ids": selected,
                "selected_count": len(selected),
                "available_count": len(node_ids),
            }
        )

        log(
            INFO,
            "configure_train: Sliding-window selected %s nodes (out of %s): %s",
            len(selected),
            len(node_ids),
            selected,
        )

        train_config = ConfigRecord(dict(config))
        train_config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: train_config})
        return list(self._construct_messages(record, selected, MessageType.TRAIN))