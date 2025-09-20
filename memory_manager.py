from typing import List, Dict, Any
from collections import deque
import json

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
    
    def add_exchange(self, user_input: str, ai_response: str, context: List[str] = None):
        """Add a user-AI exchange to memory"""
        exchange = {
            "user": user_input,
            "ai": ai_response,
            "context": context or [],
            "timestamp": self._get_timestamp()
        }
        self.conversation_history.append(exchange)
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation history for context"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in list(self.conversation_history)[-5:]:  # Last 5 exchanges
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"AI: {exchange['ai']}")
        
        return "\n".join(context_parts)
    
    def get_recent_topics(self) -> List[str]:
        """Extract recent topics/keywords from conversation"""
        topics = []
        for exchange in list(self.conversation_history)[-3:]:
            # Simple keyword extraction (you can enhance this)
            user_words = exchange['user'].lower().split()
            topics.extend([word for word in user_words if len(word) > 4])
        return list(set(topics))
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()