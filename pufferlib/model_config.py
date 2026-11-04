"""Resolve the Drive architecture selector for training and evaluation."""


def apply_model_type(config, model_type=None):
    """Explicit model selection overrides legacy policy/RNN settings in place."""
    if model_type is None:
        model_type = config.get("model_type")
    if model_type is not None:
        if model_type not in ("mlp", "lstm", "lstm_seq"):
            raise ValueError("model_type must be 'mlp', 'lstm' or 'lstm_seq'")
        if config.get("package") != "ocean" or config.get("policy_name") not in ("Drive", "DriveMLP"):
            raise ValueError("--model-type is supported only by the Drive policy")
        config["model_type"] = model_type
        config["policy_name"] = "DriveMLP" if model_type == "mlp" else "Drive"
        config["rnn_name"] = None if model_type == "mlp" else "Recurrent"
        if model_type == "lstm_seq":
            config["rnn_name"] = "RecurrentSequence"
    use_sequence = config["rnn_name"] == "RecurrentSequence"
    config["train"]["use_sequence"] = use_sequence
    config["train"]["use_rnn"] = config["rnn_name"] is not None and not use_sequence
    if use_sequence:
        length = config.get("sequence_length", 32)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("sequence_length must be a positive integer")
        config["sequence_length"] = length
        config["train"]["sequence_length"] = length
    config["train"]["model_type"] = model_type
