"""Raw observation windows shared by sequence training and all inference paths."""

import torch


class ObservationHistory:
    def __init__(self, num_agents, sequence_length, obs_shape, *, device, dtype=torch.float32):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.buffer = torch.zeros(num_agents, sequence_length, *obs_shape, device=device, dtype=dtype)
        self.initialized = torch.zeros(num_agents, device=device, dtype=torch.bool)

    def reset(self):
        self.initialized.zero_()

    @torch.no_grad()
    def append(self, observations, done=None, agent_slice=None):
        """Reset before appending the returned post-reset observation.

        agent_slice addresses stable global slots in an asynchronous vector env.
        The returned view remains valid until the next append to those slots.
        """
        if agent_slice is None:
            agent_slice = slice(None)
        window = self.buffer[agent_slice]
        initialized = self.initialized[agent_slice]
        reset = ~initialized
        if done is not None:
            reset = reset | torch.as_tensor(done, device=window.device, dtype=torch.bool)
        window[:, :-1] = window[:, 1:].clone()
        window[:, -1] = observations
        window[reset] = observations[reset, None]
        initialized.fill_(True)
        return window


def make_history(config, observations, device):
    """Create fresh history after an explicit environment reset, if requested."""
    if not config["train"].get("use_sequence", False):
        return None
    obs = torch.as_tensor(observations)
    return ObservationHistory(len(obs), config["train"]["sequence_length"], obs.shape[1:],
                              device=device, dtype=obs.dtype)


def sequence_slices(count, sequence_length, max_minibatch_size, sequence_chunks=0):
    """Partition whole windows; explicit counts produce balanced nonempty chunks.

    Zero preserves the automatic observation-budget rule. An explicit count
    overrides that rule, independently of N; at most one chunk per sample.
    """
    if isinstance(sequence_chunks, bool) or not isinstance(sequence_chunks, int) or sequence_chunks < 0:
        raise ValueError("sequence_chunks must be a non-negative integer (0 = automatic)")
    if count == 0:
        return
    if sequence_chunks:
        chunks = min(count, sequence_chunks)
        size, extra = divmod(count, chunks)
        start = 0
        for index in range(chunks):
            stop = start + size + (index < extra)
            yield slice(start, stop)
            start = stop
    else:
        size = max(1, max_minibatch_size // sequence_length)
        for start in range(0, count, size):
            yield slice(start, min(count, start + size))


def forward_with_history(policy, observations, state, history=None, done=None,
                         agent_slice=None, max_minibatch_size=32768, sequence_chunks=0):
    """Keep legacy calls unchanged; bound sequence inference activation memory."""
    if history is None:
        return policy.forward_eval(observations, state)
    windows = history.append(observations, done, agent_slice)
    parts = sequence_slices(len(windows), windows.shape[1], max_minibatch_size, sequence_chunks)
    outputs = [policy.forward_eval(windows[part], state) for part in parts]
    if len(outputs) == 1:
        return outputs[0]
    logits, values = zip(*outputs)
    if isinstance(logits[0], torch.distributions.Normal):
        logits = torch.distributions.Normal(torch.cat([x.loc for x in logits]),
                                            torch.cat([x.scale for x in logits]))
    elif isinstance(logits[0], torch.Tensor):
        logits = torch.cat(logits)
    else:
        logits = [torch.cat(head) for head in zip(*logits)]
    return logits, torch.cat(values)


def reconstruct_windows(observations, prefixes, resets, rows, times):
    """Gather only requested windows from [segments, rollout, *obs_shape].

    prefixes contains the N-1 frames before each segment's first current frame.
    resets marks a new episode BEFORE observations[row, time]. Earlier context
    in a window is replaced by that episode's first frame, never by future data.
    Indices and tensors are on the observation storage device (CPU if offloaded).
    """
    length = prefixes.shape[1] + 1
    horizon = observations.shape[1]
    steps = torch.arange(horizon, device=observations.device)
    last_reset = torch.where(resets[rows] & (steps <= times[:, None]), steps, -1).amax(dim=1)
    indices = times[:, None] + torch.arange(length, device=observations.device)
    reset_start = torch.where(last_reset >= 0, last_reset + length - 1, 0)
    indices = torch.maximum(indices, reset_start[:, None])
    current_indices = (indices - (length - 1)).clamp(min=0)
    current = observations[rows[:, None], current_indices]
    if length == 1:
        return current
    previous = prefixes[rows[:, None], indices.clamp(max=length - 2)]
    mask = (indices < length - 1).reshape(*indices.shape, *([1] * (observations.ndim - 2)))
    return torch.where(mask, previous, current)


def sequence_ppo_backward(policy, observations, prefixes, resets, rows, actions,
                          old_logprobs, old_values, returns, advantages, importance,
                          config, entropy_coef, amp_context):
    """Backpropagate one PPO minibatch without materializing all its windows.

    Normalize advantages once and weight chunk means by their sample count.
    The caller owns gradient clipping/optimizer steps at the original boundaries.
    """
    from pufferlib.pytorch import sample_logits

    shape = old_logprobs.shape
    count = old_logprobs.numel()
    horizon = shape[1]
    storage_device = observations.device
    device = old_values.device
    rows = rows.to(storage_device)
    adv = (importance * (advantages - advantages.mean()) / (advantages.std() + 1e-8)).flatten()
    flat_actions = actions.reshape(count, *actions.shape[2:])
    new_values = torch.empty_like(old_values).flatten()
    ratios = torch.empty_like(old_logprobs).flatten()
    metrics = dict.fromkeys(("policy_loss", "value_loss", "entropy", "old_approx_kl",
                            "approx_kl", "clipfrac", "importance"), 0.0)
    length = prefixes.shape[1] + 1
    parts = sequence_slices(count, length, config["max_minibatch_size"], config.get("sequence_chunks", 0))
    for part in parts:
        start, stop = part.start, part.stop
        positions = torch.arange(start, stop, device=storage_device)
        windows = reconstruct_windows(observations, prefixes, resets,
                                      rows[positions // horizon], positions % horizon).to(device)
        weight = (stop - start) / count
        with amp_context:
            logits, value = policy(windows, {})
            _, logprob, entropy = sample_logits(logits, action=flat_actions[part])
            logratio = logprob.reshape(-1) - old_logprobs.flatten()[part]
            ratio = logratio.exp()
            policy_loss = torch.maximum(-adv[part] * ratio, -adv[part] * ratio.clamp(
                1 - config["clip_coef"], 1 + config["clip_coef"])).mean()
            value = value.reshape(-1)
            old_value = old_values.flatten()[part]
            clipped = old_value + (value - old_value).clamp(-config["vf_clip_coef"], config["vf_clip_coef"])
            target = returns.flatten()[part]
            value_loss = 0.5 * torch.maximum((value - target).square(), (clipped - target).square()).mean()
            entropy_loss = entropy.mean()
            loss = (policy_loss + config["vf_coef"] * value_loss - entropy_coef * entropy_loss) * weight
        loss.backward()
        with torch.no_grad():
            new_values[part] = value.float()
            ratios[part] = ratio
            chunk_metrics = dict(policy_loss=policy_loss, value_loss=value_loss, entropy=entropy_loss,
                                 old_approx_kl=(-logratio).mean(),
                                 approx_kl=((ratio - 1) - logratio).mean(),
                                 clipfrac=((ratio - 1).abs() > config["clip_coef"]).float().mean(),
                                 importance=ratio.mean())
            for name, metric in chunk_metrics.items():
                metrics[name] += metric.item() * weight
        # Release the input before constructing the next chunk's windows.
        del windows, logits, loss
    return metrics, new_values.reshape(shape), ratios.reshape(shape)
