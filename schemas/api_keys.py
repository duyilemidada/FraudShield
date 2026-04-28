from pydantic import BaseModel
from pydantic import ConfigDict
class APIKeyCreate(BaseModel):
    label: str

class APIKeyResponse(BaseModel):
    id: int
    label: str
    key: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)