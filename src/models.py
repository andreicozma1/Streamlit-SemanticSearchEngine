model_context_windows = {
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-1106": 16384,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
}


model_pricing = {
    "gpt-4-1106-preview": dict(
        input=0.01,
        output=0.03,
        tokens=1000,
    ),
    "gpt-4-vision-preview": dict(
        input=0.01,
        output=0.03,
        tokens=1000,
    ),
    "gpt-4": dict(
        input=0.03,
        output=0.06,
        tokens=1000,
    ),
    "gpt-4-32k": dict(
        input=0.06,
        output=0.12,
        tokens=1000,
    ),
    "gpt-4-0613": dict(
        input=0.03,
        output=0.06,
        tokens=1000,
    ),
    "gpt-4-32k-0613": dict(
        input=0.06,
        output=0.12,
        tokens=1000,
    ),
    "gpt-3.5-turbo-1106": dict(
        input=0.0010,
        output=0.0020,
        tokens=1000,
    ),
    "gpt-3.5-turbo": dict(
        input=0.0015,
        output=0.0020,
        tokens=1000,
    ),
    "gpt-3.5-turbo-16k": dict(
        input=0.0010,
        output=0.0020,
        tokens=1000,
    ),
}

assert (
    model_context_windows.keys() == model_pricing.keys()
), "Internal error. Model keys are not the same."

model_list = list(model_context_windows.keys())
