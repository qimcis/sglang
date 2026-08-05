"""Architecture-neutral protection for persistent inference state."""

from sglang.srt.state_protection.manager import (
    StateProtectionManager,
    install_state_protection,
)

__all__ = ["StateProtectionManager", "install_state_protection"]

