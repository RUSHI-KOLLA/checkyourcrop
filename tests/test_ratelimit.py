import time

from app.ratelimit import RateLimiter


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(requests_per_window=5, window_seconds=60)
        for _ in range(5):
            allowed, retry = rl.is_allowed("127.0.0.1")
            assert allowed is True
            assert retry == 0

    def test_blocks_over_limit(self):
        rl = RateLimiter(requests_per_window=5, window_seconds=60)
        for _ in range(5):
            rl.is_allowed("127.0.0.1")
        allowed, retry = rl.is_allowed("127.0.0.1")
        assert allowed is False
        assert retry > 0

    def test_separate_ips(self):
        rl = RateLimiter(requests_per_window=2, window_seconds=60)
        rl.is_allowed("127.0.0.1")
        rl.is_allowed("127.0.0.1")
        # First IP blocked
        allowed, _ = rl.is_allowed("127.0.0.1")
        assert allowed is False
        # Second IP still allowed
        allowed, _ = rl.is_allowed("127.0.0.2")
        assert allowed is True

    def test_window_expiry(self):
        rl = RateLimiter(requests_per_window=2, window_seconds=1)
        rl.is_allowed("127.0.0.1")
        rl.is_allowed("127.0.0.1")
        allowed, _ = rl.is_allowed("127.0.0.1")
        assert allowed is False
        # Wait for window to expire
        time.sleep(1.1)
        allowed, _ = rl.is_allowed("127.0.0.1")
        assert allowed is True

    def test_get_remaining(self):
        rl = RateLimiter(requests_per_window=5, window_seconds=60)
        assert rl.get_remaining("127.0.0.1") == 5
        rl.is_allowed("127.0.0.1")
        assert rl.get_remaining("127.0.0.1") == 4
        rl.is_allowed("127.0.0.1")
        assert rl.get_remaining("127.0.0.1") == 3

    def test_get_remaining_after_expiry(self):
        rl = RateLimiter(requests_per_window=2, window_seconds=1)
        rl.is_allowed("127.0.0.1")
        rl.is_allowed("127.0.0.1")
        assert rl.get_remaining("127.0.0.1") == 0
        time.sleep(1.1)
        assert rl.get_remaining("127.0.0.1") == 2
