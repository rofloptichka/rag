from pydantic import BaseModel

# ------------------------------------------------------------------------------
# Pydantic models for request bodies
# ------------------------------------------------------------------------------
class UpdateSendableDescription(BaseModel):
    new_description: str
