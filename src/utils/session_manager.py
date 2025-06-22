import aiohttp


class SessionManager:
    _instance = None

    def __init__(self):
        if SessionManager._instance is None:
            self._session = None
            SessionManager._instance = self

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session is not None:
            await self._session.close()
            self._session = None


session_manager = SessionManager.get_instance()
