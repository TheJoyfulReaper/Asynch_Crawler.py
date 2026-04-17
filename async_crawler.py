"""
async_crawler.py — Asynchronous Web Crawler with Rate Limiting & Retry Logic

Demonstrates:
  - asyncio / aiohttp for concurrent HTTP requests
  - Token-bucket rate limiter (thread-safe, async-aware)
  - Exponential backoff with jitter for transient failures
  - Producer/consumer pattern via asyncio.Queue
  - Structured logging and graceful shutdown
  - Type hints and dataclass usage throughout

Author: Christopher Hall
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crawler")


class CrawlStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


@dataclass
class CrawlResult:
    url: str
    status: CrawlStatus
    status_code: Optional[int] = None
    content_hash: Optional[str] = None
    links_found: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CrawlerConfig:
    """Centralised configuration — every tuneable lives here."""
    seed_urls: list[str] = field(default_factory=list)
    max_depth: int = 3
    max_pages: int = 100
    concurrency: int = 10
    requests_per_second: float = 5.0
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 30.0
    request_timeout: float = 15.0
    allowed_domains: set[str] = field(default_factory=set)
    user_agent: str = "SeniorEngBot/1.0 (portfolio-demo)"


# ---------------------------------------------------------------------------
# Token-Bucket Rate Limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """
    Async-compatible token-bucket rate limiter.

    Tokens refill at a steady rate.  Each request consumes one token.
    If the bucket is empty the caller awaits until a token is available.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        self._rate = rate                       # tokens per second
        self._burst = burst or int(rate * 2)    # max stored tokens
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a token is available; return wait time in seconds."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            # Calculate how long until the next token arrives
            deficit = 1.0 - self._tokens
            wait = deficit / self._rate
        await asyncio.sleep(wait)
        async with self._lock:
            self._refill()
            self._tokens -= 1.0
        return wait

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Retry with Exponential Backoff + Jitter
# ---------------------------------------------------------------------------

async def retry_with_backoff(
    coro_factory,
    *,
    max_retries: int = 3,
    base: float = 0.5,
    cap: float = 30.0,
    retryable: tuple[type[Exception], ...] = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
    ),
):
    """
    Retry an async callable with decorrelated-jitter backoff.

    Uses the "full jitter" algorithm from the AWS Architecture Blog:
        sleep = random(0, min(cap, base * 2 ** attempt))
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = random.uniform(0, min(cap, base * (2 ** attempt)))
            logger.warning(
                "Attempt %d failed (%s). Retrying in %.2fs…",
                attempt + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Crawler Core
# ---------------------------------------------------------------------------

class AsyncCrawler:
    """
    A production-style async web crawler.

    Architecture:
        1.  A *frontier queue* (asyncio.Queue) holds (url, depth) pairs.
        2.  N worker coroutines pull from the queue, fetch pages, extract
            links, and push newly discovered URLs back onto the frontier.
        3.  A TokenBucketRateLimiter gates outbound requests.
        4.  A seen-set (by normalised URL) prevents duplicate work.
        5.  Graceful shutdown via asyncio.Event.
    """

    def __init__(self, config: CrawlerConfig) -> None:
        self.cfg = config
        self._frontier: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self._seen: set[str] = set()
        self._results: list[CrawlResult] = []
        self._limiter = TokenBucketRateLimiter(rate=config.requests_per_second)
        self._shutdown = asyncio.Event()
        self._pages_crawled = 0
        self._lock = asyncio.Lock()

    # -- public API --

    async def run(self) -> list[CrawlResult]:
        """Seed the frontier and run workers until completion."""
        for url in self.cfg.seed_urls:
            self._enqueue(url, depth=0)

        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
        connector = aiohttp.TCPConnector(limit=self.cfg.concurrency)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": self.cfg.user_agent},
        ) as session:
            workers = [
                asyncio.create_task(self._worker(session, i))
                for i in range(self.cfg.concurrency)
            ]
            # Wait until the queue drains and all workers are idle
            await self._frontier.join()
            self._shutdown.set()
            await asyncio.gather(*workers)

        logger.info(
            "Crawl complete — %d pages fetched, %d results total.",
            self._pages_crawled,
            len(self._results),
        )
        return self._results

    # -- internals --

    def _enqueue(self, url: str, depth: int) -> bool:
        normalised = self._normalise(url)
        if normalised in self._seen:
            return False
        if self.cfg.allowed_domains:
            domain = urlparse(normalised).netloc
            if domain not in self.cfg.allowed_domains:
                return False
        self._seen.add(normalised)
        self._frontier.put_nowait((normalised, depth))
        return True

    async def _worker(self, session: aiohttp.ClientSession, wid: int) -> None:
        """Worker loop — pull URLs, fetch, extract links, repeat."""
        while not self._shutdown.is_set():
            try:
                url, depth = await asyncio.wait_for(
                    self._frontier.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                if self._frontier.empty():
                    break
                continue

            async with self._lock:
                if self._pages_crawled >= self.cfg.max_pages:
                    self._frontier.task_done()
                    continue

            result = await self._fetch(session, url, depth)
            self._results.append(result)

            async with self._lock:
                self._pages_crawled += 1

            self._frontier.task_done()

    async def _fetch(
        self, session: aiohttp.ClientSession, url: str, depth: int
    ) -> CrawlResult:
        """Fetch a single URL with rate limiting and retry."""
        wait = await self._limiter.acquire()
        if wait > 0:
            logger.debug("Rate-limited %.2fs before fetching %s", wait, url)

        start = time.monotonic()
        try:
            response = await retry_with_backoff(
                lambda: session.get(url),
                max_retries=self.cfg.max_retries,
                base=self.cfg.backoff_base,
                cap=self.cfg.backoff_max,
            )
            async with response:
                body = await response.text(errors="replace")
                elapsed = (time.monotonic() - start) * 1000

                content_hash = hashlib.sha256(body.encode()).hexdigest()[:16]
                links = self._extract_links(body, url)

                if depth < self.cfg.max_depth:
                    for link in links:
                        self._enqueue(link, depth + 1)

                logger.info(
                    "✓ [%d] %s — %d links (%.0fms)",
                    response.status,
                    url,
                    len(links),
                    elapsed,
                )
                return CrawlResult(
                    url=url,
                    status=CrawlStatus.SUCCESS,
                    status_code=response.status,
                    content_hash=content_hash,
                    links_found=len(links),
                    elapsed_ms=elapsed,
                )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("✗ %s — %s (%.0fms)", url, exc, elapsed)
            return CrawlResult(
                url=url,
                status=CrawlStatus.FAILED,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    @staticmethod
    def _extract_links(html: str, base_url: str) -> list[str]:
        """Naive link extractor — keeps it dependency-light for demo."""
        import re

        links: list[str] = []
        for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
            href = match.group(1)
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in ("http", "https"):
                # Strip fragments
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean += f"?{parsed.query}"
                links.append(clean)
        return links

    @staticmethod
    def _normalise(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

async def main() -> None:
    config = CrawlerConfig(
        seed_urls=["https://example.com"],
        max_depth=2,
        max_pages=50,
        concurrency=5,
        requests_per_second=3.0,
        allowed_domains={"example.com"},
    )

    crawler = AsyncCrawler(config)
    results = await crawler.run()

    # Summary statistics
    successes = [r for r in results if r.status == CrawlStatus.SUCCESS]
    failures = [r for r in results if r.status == CrawlStatus.FAILED]
    avg_latency = (
        sum(r.elapsed_ms for r in successes) / len(successes)
        if successes
        else 0
    )

    print("\n" + "=" * 60)
    print(f"  Crawl Summary")
    print(f"  Pages fetched : {len(successes)}")
    print(f"  Failures      : {len(failures)}")
    print(f"  Avg latency   : {avg_latency:.1f}ms")
    print(f"  Total links   : {sum(r.links_found for r in successes)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
