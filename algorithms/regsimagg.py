"""RegSimAgg Strategy: Sample-size + Model-similarity aggregation with Temporal Damping."""

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, Message
from flwr.common import MetricRecord
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

_EPS = 1e-12
_SIM_EPS = 1e-5

def _state_from_reply(reply: Message, arrayrecord_key: str) -> OrderedDict[str, np.ndarray]:
    """Extract model weights from reply message into an OrderedDict of NumPy arrays."""
    return OrderedDict(
        (name, tensor.detach().cpu().numpy().copy())
        for name, tensor in reply.content[arrayrecord_key].to_torch_state_dict().items()
    )


def _float_vector(state: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all floating point parameters into a single 1D vector."""
    parts = [
        v.astype(np.float64, copy=False).ravel()
        for v in state.values()
        if np.issubdtype(v.dtype, np.floating)
    ]
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)


def _similarity_weights(
    states: list[dict[str, np.ndarray]], mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate model-similarity weights based on inverse L1 distance to centroid."""
    vectors = [_float_vector(state) for state in states]
    if mode == "github_compat":
        totals = np.asarray([v.sum() for v in vectors], dtype=np.float64)
        distances = np.abs(totals - totals.mean())
    else:  # "paper_l1"
        centroid = np.mean(np.stack(vectors), axis=0)
        distances = np.asarray([np.sum(np.abs(v - centroid)) for v in vectors],
            dtype=np.float64,)

    similarity = 1.0 / (distances + _SIM_EPS)
    similarity_weights = similarity / similarity.sum()
    return similarity_weights, distances

def _aggregate_states(
    states: list[dict[str, np.ndarray]], weights: np.ndarray
) -> OrderedDict[str, torch.Tensor]:
    """Weighted aggregation of parameter arrays into PyTorch tensors."""
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, first in states[0].items():
        if np.issubdtype(first.dtype, np.floating):
            value = sum(
                weight * state[name].astype(np.float64, copy=False)
                for state, weight in zip(states, weights)
            )
            result[name] = torch.from_numpy(value.astype(first.dtype, copy=False))
        else:
            result[name] = torch.from_numpy(first.copy())
    return result


class RegSimAggStrategy(FedAvg):
    """RegSimAgg: Sample-size + model-similarity aggregation with temporal damping."""

    def __init__(
        self,
        *,
        regularization_round: int = 10,
        distance_mode: str = "paper_l1",
        weighted_by_key: str = "num-examples",
        **kwargs: Any,
    ) -> None:
        if regularization_round < 0:
            raise ValueError("regularization_round must be non-negative")
        if distance_mode not in {"paper_l1", "github_compat"}:
            raise ValueError("distance_mode must be 'paper_l1' or 'github_compat'")

        super().__init__(weighted_by_key=weighted_by_key, **kwargs)
        self.regularization_round = regularization_round
        self.distance_mode = distance_mode
        self.weighted_by_key = weighted_by_key
        self.previous_global: OrderedDict[str, np.ndarray] | None = None
        self.aggregation_audit: list[dict[str, Any]] = []

    def aggregate_train(
        self, server_round: int, replies: list[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, MetricRecord()

        institution_ids = [
            int(reply.content["client-info"]["institution-id"])
            for reply in valid_replies
        ]

        states = [_state_from_reply(reply, self.arrayrecord_key) for reply in valid_replies]
        counts = np.asarray(
            [float(reply.content["metrics"][self.weighted_by_key]) for reply in valid_replies],
            dtype=np.float64,
        )

        # 1. Sample-size weights
        sample_weights = counts / max(counts.sum(), _EPS)

        # 2. Model similarity weights
        similarity_weights, distances = _similarity_weights(states, self.distance_mode)

        # 3. Combined weight (50% sample size, 50% model similarity)
        final_weights = 0.5 * (sample_weights + similarity_weights)

        # 4. Temporal regularization damping
        temporal_change = np.full(len(states), np.nan)
        if server_round > self.regularization_round and self.previous_global is not None:
            temporal_change = np.asarray(
                [np.mean(np.abs(_float_vector(s) - _float_vector(self.previous_global))) for s in states]
            )
            final_weights = final_weights / (temporal_change + _EPS)

        final_weights = final_weights / final_weights.sum()

        # 5. Weighted state aggregation
        aggregated_state = _aggregate_states(states, final_weights)
        self.previous_global = OrderedDict(
            (n, v.detach().cpu().numpy().copy()) for n, v in aggregated_state.items()
        )

        # Research audit trail
        self.aggregation_audit.append(
            {
                "round": server_round,
                "node_ids": [str(r.metadata.src_node_id) for r in valid_replies],
                "num_examples": counts.tolist(),
                "sample_weights": sample_weights.tolist(),
                "distances": distances.tolist(),
                "similarity_weights": similarity_weights.tolist(),
                "temporal_change": temporal_change.tolist(),
                "aggregation_weights": final_weights.tolist(),
                "institution_ids": institution_ids,
            }
        )

        return ArrayRecord(aggregated_state), aggregate_metricrecords(
            [reply.content for reply in valid_replies], self.weighted_by_key
        )
