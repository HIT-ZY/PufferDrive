"""Resolve the Drive architecture selector for training and evaluation."""


def apply_model_type(config, model_type=None):
    """Explicit model selection overrides legacy policy/RNN settings in place."""
    if model_type is None:
        model_type = config.get("model_type")
    if model_type is not None:
        if model_type not in ("mlp", "lstm"):
            raise ValueError("model_type must be 'mlp' or 'lstm'")
        if config.get("package") != "ocean" or config.get("policy_name") not in ("Drive", "DriveMLP"):
            raise ValueError("--model-type is supported only by the Drive policy")
        config["model_type"] = model_type
        config["policy_name"] = "DriveMLP" if model_type == "mlp" else "Drive"
        config["rnn_name"] = None if model_type == "mlp" else "Recurrent"
    config["train"]["use_rnn"] = config["rnn_name"] is not None
