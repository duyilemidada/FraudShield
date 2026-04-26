from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

client = AsyncIOMotorClient(settings.MONGODB_URL)
database = client["fraudshield"]
transaction_collection = database["transactions"]