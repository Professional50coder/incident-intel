"""Project-wide constants. Import from here instead of hardcoding values that must stay
consistent across data loading, training, and serving (e.g. frame size)."""

FRAME_SIZE: tuple[int, int] = (64, 64)  # (H, W) - downsized from Ped2's native 240x360 to
                                          # keep training comfortably inside 4GB VRAM.
