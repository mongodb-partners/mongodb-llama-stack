# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import MongoDBIOConfig, SearchMode, PipelineConfig, GraphLookupConfig

__all__ = [
    "MongoDBIOConfig",
    "SearchMode", 
    "PipelineConfig",
    "GraphLookupConfig",
    "get_adapter_impl"
]


async def get_adapter_impl(config: MongoDBIOConfig, deps: Dict[Api, ProviderSpec]):
    from .mongodb import MongoDBIOAdapter      

    impl = MongoDBIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl