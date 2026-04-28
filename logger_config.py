import logging
from uvicorn.logging import ColourizedFormatter
from logging.handlers import TimedRotatingFileHandler
import os

def set_client_logger():
  logger = logging.getLogger("client.logger")
  logger.setLevel(logging.INFO)

  #console handler
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(
    ColourizedFormatter("%(levelprefix)s CLIENT CALL - %(message)s", use_colors=True)
  )

  logger.addHandler(console_handler)

  if not os.getenv("TESTING"): 

    #File handler
    file_handler = TimedRotatingFileHandler("app.log", delay=True)
    file_handler.setFormatter(
      logging.Formatter("time %(asctime)s, %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

  return logger

client_logger = set_client_logger()