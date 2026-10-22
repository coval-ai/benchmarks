# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""STT provider registry.

Every public module in this package defines exactly one :class:`STTProvider`
subclass and is registered under its module name, so adding a provider is
adding a file. Usage::

    from coval_bench.providers.stt import STT_PROVIDERS

    provider = STT_PROVIDERS["deepgram"](settings, model="nova-3")
    result = await provider.measure_ttft(...)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from coval_bench.providers.base import STTProvider


def _discover() -> dict[str, type[STTProvider]]:
    registry: dict[str, type[STTProvider]] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__ == module.__name__
            and issubclass(cls, STTProvider)
            and not inspect.isabstract(cls)
        ]
        if len(classes) != 1:
            raise RuntimeError(
                f"{module.__name__} must define exactly one STTProvider subclass, "
                f"found {[c.__name__ for c in classes]}"
            )
        registry[module_info.name] = classes[0]
    return registry


STT_PROVIDERS: dict[str, type[STTProvider]] = _discover()

__all__ = ["STT_PROVIDERS"]
