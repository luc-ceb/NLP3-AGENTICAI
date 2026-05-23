import asyncio
from src.agent.agent import *
from uuid import uuid4
if __name__ == "__main__":
    session_id = str(uuid4())
    asyncio.run(interactive(session_id=session_id))