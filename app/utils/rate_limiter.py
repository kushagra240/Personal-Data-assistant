import time

from fastapi import HTTPException, Request, status

from app.utils.logger import logger


class RateLimiter:
    def __init__(self, requests_limit: int = 30, window_seconds: int = 60):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.client_records = {}  # client_ip -> list of timestamps

    def check_rate_limit(self, request: Request):
        """Enforces sliding window rate limiting based on client IP."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Initialize client record
        if client_ip not in self.client_records:
            self.client_records[client_ip] = []

        # Filter timestamps outside the sliding window
        self.client_records[client_ip] = [t for t in self.client_records[client_ip] if now - t < self.window_seconds]

        # Check limit
        if len(self.client_records[client_ip]) >= self.requests_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Please try again later."
            )

        # Record request
        self.client_records[client_ip].append(now)


# Instantiate rate limiters: 10 uploads/min, 30 questions/min
upload_rate_limiter = RateLimiter(requests_limit=10, window_seconds=60)
message_rate_limiter = RateLimiter(requests_limit=30, window_seconds=60)
