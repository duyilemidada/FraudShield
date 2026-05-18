import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger('uvicorn.error')
SLA_THRESHOLD_MS = 500

class RequestTimingMiddleware(BaseHTTPMiddleware):
  async def dispatch(self, request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    ms = (time.perf_counter() - start) * 1000

    # Add timing header so merchants can see latency in responses
    response.headers['X-Response-Time-Ms'] = f'{ms:.1f}'

    if '/predict' in request.url.path:
      if ms > SLA_THRESHOLD_MS:
        logger.warning(f'SLOW PREDICT: {ms:.0f}ms (SLA={SLA_THRESHOLD_MS}ms)')
      else :
        logger.info(f'predict: {ms:.0f}ms')

    return response