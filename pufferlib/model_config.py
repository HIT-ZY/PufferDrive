"""Resolve the Drive architecture selector for training and evaluation."""


TRANSFORMER_DEFAULTS = dict(num_layers=1, num_heads=4, dim_feedforward=512)


def transformer_settings(hidden_size, settings=None):
    """Normalize and validate the temporal encoder dimensions without importing torch."""
    settings = {**TRANSFORMER_DEFAULTS, **(settings or {})}
    for name, value in dict(hidden_size=hidden_size, **settings).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"transformer {name} must be a positive integer")
    if hidden_size % settings["num_heads"]:
        raise ValueError("policy.hidden_size must be divisible by transformer.num_heads")
    return settings


def wrapper_kwargs(config):
    """Keep training and evaluation construction of temporal wrappers identical."""
    if config["rnn_name"] == "TransformerSequence":
        return dict(config["transformer"], sequence_length=config["sequence_length"])
    kwargs = dict(config["rnn"])
    if config["train"].get("use_sequence", False):
        kwargs["sequence_length"] = config["sequence_length"]
    return kwargs


def apply_model_type(config, model_type=None):
    """Explicit model selection overrides legacy policy/RNN settings in place."""
    if model_type is None:
        model_type = config.get("model_type")
    if model_type is None and config["rnn_name"] == "TransformerSequence":
        model_type = "transformer"
    if model_type is not None:
        if model_type not in ("mlp", "lstm", "lstm_seq", "transformer"):
            raise ValueError("model_type must be 'mlp', 'lstm', 'lstm_seq' or 'transformer'")
        if config.get("package") != "ocean" or config.get("policy_name") not in ("Drive", "DriveMLP"):
            raise ValueError("--model-type is supported only by the Drive policy")
        config["model_type"] = model_type
        config["policy_name"] = "DriveMLP" if model_type == "mlp" else "Drive"
        config["rnn_name"] = None if model_type == "mlp" else "Recurrent"
        if model_type == "lstm_seq":
            config["rnn_name"] = "RecurrentSequence"
        elif model_type == "transformer":
            config["rnn_name"] = "TransformerSequence"
            width = config["policy"].get("hidden_size", 128)
            config["transformer"] = transformer_settings(width, config.get("transformer"))
            config["train"]["transformer"] = dict(config["transformer"], hidden_size=width)
    use_sequence = config["rnn_name"] in ("RecurrentSequence", "TransformerSequence")
    config["train"]["use_sequence"] = use_sequence
    config["train"]["use_rnn"] = config["rnn_name"] is not None and not use_sequence
    if use_sequence:
        length = config.get("sequence_length", 32)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("sequence_length must be a positive integer")
        config["sequence_length"] = length
        config["train"]["sequence_length"] = length
        chunks = config.get("sequence_chunks", 0)
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 0:
            raise ValueError("sequence_chunks must be a non-negative integer (0 = automatic)")
        config["sequence_chunks"] = chunks
        config["train"]["sequence_chunks"] = chunks
    config["train"]["model_type"] = model_type
