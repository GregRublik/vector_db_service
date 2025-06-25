from aiohttp import ClientSession
from config.settings import SessionManager
from fastapi import Depends

def get_session(
        http_session: ClientSession = Depends(SessionManager.get_session),
) -> ClientSession:
    return http_session
