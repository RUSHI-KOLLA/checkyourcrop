import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    requests_per_window: int
    window_seconds: int
    _requests: dict[str, list] = field(default_factory=lambda: defaultdict(list))

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, retry_after_seconds)."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._requests[client_ip] = [ts for ts in self._requests[client_ip] if ts > window_start]

        if len(self._requests[client_ip]) >= self.requests_per_window:
            # Find when the oldest request expires
            oldest = self._requests[client_ip][0]
            retry_after = int(oldest + self.window_seconds - now) + 1
            return False, max(retry_after, 1)

        self._requests[client_ip].append(now)
        return True, 0

    def get_remaining(self, client_ip: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[client_ip] = [ts for ts in self._requests[client_ip] if ts > window_start]
        return max(0, self.requests_per_window - len(self._requests[client_ip]))
