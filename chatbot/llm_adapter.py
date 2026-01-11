"""
LLM Adapter module - provides a unified interface for different LLM providers.
"""
from __future__ import annotations

import os
import time
import configparser
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    model: str
    temperature: float = 0.4
    max_retries: int = 5
    sleep_between_calls: float = 0.3


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            system_prompt: System message to set context/behavior
            user_prompt: User message with the actual prompt
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """OpenAI LLM adapter using the OpenAI API."""
    
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using OpenAI's API with retry logic."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                )
                
                text = (response.choices[0].message.content or "").strip()
                                
                return text
                
            except Exception as exc:
                wait = min(2 ** attempt, 20)
                print(f"[WARN] Attempt {attempt}/{self.config.max_retries} failed: {exc}")
                if attempt < self.config.max_retries:
                    time.sleep(wait)
        
        raise RuntimeError(f"LLM generation failed after {self.config.max_retries} retries")


class HuggingFaceAdapter(LLMAdapter):
    """Hugging Face LLM adapter (placeholder for future implementation)."""
    
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            "Hugging Face adapter is not yet implemented. "
            "Use 'openai' provider for now."
        )
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Hugging Face adapter not yet implemented")


# Factory function to create the appropriate adapter
def create_llm_adapter(
    provider: Literal["openai", "huggingface"] = "openai",
    model: str | None = None,
    temperature: float = 0.4,
    max_retries: int = 5,
    sleep_between_calls: float = 0.3,
) -> LLMAdapter:
    """
    Factory function to create an LLM adapter.
    
    Args:
        provider: LLM provider to use ('openai' or 'huggingface')
        model: Model name (defaults based on provider and config.ini)
        temperature: Sampling temperature for generation
        max_retries: Maximum number of retry attempts
        sleep_between_calls: Sleep duration between API calls
        
    Returns:
        Configured LLM adapter instance
        
    Raises:
        ValueError: If provider is not supported
    """
    # Set default model based on provider
    if model is None:
        if provider == "openai":
            # Try to read from config.ini
            try:
                config = configparser.ConfigParser()
                config_path = Path(__file__).resolve().parent / "config.ini"
                if config_path.exists():
                    config.read(config_path)
                    model = config.get("AI", "llm_model_default", fallback="gpt-3.5-turbo")
                else:
                    model = "gpt-3.5-turbo"
            except Exception:
                model = "gpt-3.5-turbo"
        elif provider == "huggingface":
            model = "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        sleep_between_calls=sleep_between_calls,
    )
    
    if provider == "openai":
        return OpenAIAdapter(config)
    elif provider == "huggingface":
        return HuggingFaceAdapter(config)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'huggingface'")


if __name__ == "__main__":
    # Simple test for the OpenAI adapter
    adapter = create_llm_adapter(provider="openai")
    
    system_prompt = "You are a helpful assistant. You reply in max 200 words."
    user_prompt = "Say 'Hello, World!' and explain what it means in programming."
    
    try:
        response = adapter.generate(system_prompt, user_prompt)
        print("Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
