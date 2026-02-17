def unfreeze_last_swin_stages(model, num_stages: int = 2):
    print(f">> Unfreezing last {num_stages} Swin blocks for fine-tuning...")

    # Swin-B has layers.0, .1, .2, .3
    stage_names = [f"layers.{i}" for i in range(4 - num_stages, 4)]

    for name, param in model.vision_encoder.named_parameters():
        if any(s in name for s in stage_names):
            param.requires_grad = True