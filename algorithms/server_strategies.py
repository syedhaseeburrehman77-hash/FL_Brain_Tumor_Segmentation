from flwr.serverapp.strategy import FedAvg, FedProx
from .collaborator_selector import SlidingWindowTrainMixin
from .regsimagg import RegSimAggStrategy

STRATEGY_REGISTRY = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "regsimagg": RegSimAggStrategy,
}

def with_sliding_window(strategy_cls):
    """Dynamically compose any strategy class with SlidingWindowTrainMixin."""
    class SlidingStrategy(SlidingWindowTrainMixin, strategy_cls):
        pass
    SlidingStrategy.__name__ = f"Sliding{strategy_cls.__name__}"
    return SlidingStrategy

def get_strategy(algorithm: str, context=None, **kwargs):
    cfg = context.run_config if context is not None else {}
    strategy_kwargs = dict(kwargs)

 
    if algorithm == "fedprox" and "proximal_mu" not in strategy_kwargs:
        strategy_kwargs["proximal_mu"] = float(cfg.get("proximal_mu", 0.01))
    elif algorithm == "regsimagg":
        strategy_kwargs.setdefault("regularization_round", int(cfg.get("regsimagg-regularization-round", 10)))
        strategy_kwargs.setdefault("distance_mode", str(cfg.get("regsimagg-distance-mode", "paper_l1")))


    collaborator_selector = str(cfg.get("collaborator-selector", "fixed")).lower()
    if collaborator_selector == "sliding":
        strategy_kwargs.setdefault("collaborator_selection_fraction", float(cfg.get("collaborator-fraction", 0.2)))
        strategy_kwargs.setdefault("collaborator_selection_seed", int(cfg.get("collaborator-selector-seed", 42)))
    try:
        strategy_cls = STRATEGY_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    
    # Dynamically wrap with sliding window only if collaborator-selector is 'sliding'
    if collaborator_selector == "sliding":
        strategy_cls = with_sliding_window(strategy_cls)
    return strategy_cls(**strategy_kwargs)
