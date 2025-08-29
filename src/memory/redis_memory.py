from typing import Dict, List, Optional
import json
import redis
from datetime import datetime, timedelta

class RedisMemory:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.session_ttl = timedelta(hours=24)  # Sessions expire after 24 hours

    def save_context(self, session_id: str, context: Dict) -> None:
        """Save conversation context to Redis"""
        key = f"session:{session_id}"
        self.redis_client.set(
            key,
            json.dumps(context),
            ex=self.session_ttl
        )

    def get_context(self, session_id: str) -> Optional[Dict]:
        """Retrieve conversation context from Redis"""
        key = f"session:{session_id}"
        context = self.redis_client.get(key)
        if context:
            return json.loads(context)
        return None

    def update_context(self, session_id: str, new_data: Dict) -> None:
        """Update existing context with new data"""
        context = self.get_context(session_id) or {}
        context.update(new_data)
        self.save_context(session_id, context)

    def save_patient_info(self, session_id: str, info: Dict) -> None:
        """Save patient information"""
        key = f"patient:{session_id}"
        self.redis_client.set(
            key,
            json.dumps(info),
            ex=self.session_ttl
        )

    def get_patient_info(self, session_id: str) -> Optional[Dict]:
        """Retrieve patient information"""
        key = f"patient:{session_id}"
        info = self.redis_client.get(key)
        if info:
            return json.loads(info)
        return None
