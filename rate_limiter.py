from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request


def get_api_key_from_request(request: Request) -> str:
  """
    Extract the API key from the X-API-KEY header.
    Each API key gets its own rate limit bucket.
    If no key present (public endpoints), fall back to IP.
  """
  api_key = request.headers.get('X-API-KEY')
  if api_key :
    return api_key # limit per API key
  return get_remote_address(request) # fallback: limit per IP
  

limiter = Limiter(key_func=get_api_key_from_request)