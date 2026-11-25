def filter_compatible_state_dict(model, checkpoint_state):
    current_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in checkpoint_state.items():
        current_value = current_state.get(key)
        if current_value is None:
            compatible_state[key] = value
            continue
        if current_value.shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(current_value.shape)))

    return compatible_state, skipped


def print_skipped_checkpoint_keys(skipped, max_items=20):
    if not skipped:
        return

    print(f"Skipped {len(skipped)} checkpoint parameter(s) with incompatible shapes:")
    for key, checkpoint_shape, model_shape in skipped[:max_items]:
        print(f"  {key}: checkpoint={checkpoint_shape}, model={model_shape}")
    if len(skipped) > max_items:
        print(f"  ... {len(skipped) - max_items} more")
