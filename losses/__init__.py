from .bce import build_loss as build_bce
from .dice import build_loss as build_dice
from .bce_dice import build_loss as build_bce_dice
from .focal import build_loss as build_focal
from .tversky import build_loss as build_tversky
from .hard_afm import build_loss as build_hard_afm
import inspect

def get_loss(loss_name, num_classes=1):
    all_losses = {
        'bce': build_bce,
        'dice': build_dice,
        'bce_dice': build_bce_dice,
        'focal': build_focal,
        'tversky': build_tversky,
        'hard_afm': build_hard_afm,
    }

    if loss_name not in all_losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(all_losses.keys())}")

    if num_classes == 1:
        print(f"✓ Using {loss_name.upper()} for BINARY")
    else:
        if loss_name == 'bce':
            print("⚠️  Warning: BCE loss với multi-class - recommend CE/Dice")
        print(f"✓ Using {loss_name.upper()} for {num_classes}-class")

    build_fn = all_losses[loss_name]
    sig = inspect.signature(build_fn)

    if 'num_classes' in sig.parameters:
        return build_fn(num_classes=num_classes)
    else:
        print(f"⚠️  {loss_name} không hỗ trợ num_classes, dùng default")
        return build_fn()