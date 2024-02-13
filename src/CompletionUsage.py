from typing import Optional


from . import models


class CompletionUsage:
    def __init__(
        self,
        completion_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens_price: int = 0,
        prompt_tokens_price: int = 0,
        model: Optional[str] = None,
        **kwargs,
    ):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens_price = completion_tokens_price
        self.prompt_tokens_price = prompt_tokens_price
        if model is not None:
            model_pricing_dict = models.model_pricing[model]
            price_input = model_pricing_dict["input"]
            price_output = model_pricing_dict["output"]
            price_tokens = model_pricing_dict["tokens"]
            # the input and output price is per price_tokens tokens
            # prompt is input
            # completion is output
            self.completion_tokens_price = price_output * (
                completion_tokens / price_tokens
            )
            self.prompt_tokens_price = price_input * (prompt_tokens / price_tokens)

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_tokens_price(self):
        return self.prompt_tokens_price + self.completion_tokens_price

    # allow summing
    def __add__(self, other):
        return CompletionUsage(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens_price=self.completion_tokens_price
            + other.completion_tokens_price,
            prompt_tokens_price=self.prompt_tokens_price + other.prompt_tokens_price,
        )

    def __str__(self):
        return f"CompletionUsage(completion_tokens={self.completion_tokens} (${self.completion_tokens_price}), prompt_tokens={self.prompt_tokens} (${self.prompt_tokens_price}), total_tokens={self.total_tokens} (${self.total_tokens_price}))"
