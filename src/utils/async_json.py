import json
import os
import re
from typing import Dict, List, Any

import aiofiles


async def json_load(path: str) -> Any | List[Any] | Dict:
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        text = await f.read()
    return json.loads(text)


async def json_dump(path: str, o: List[Any] | Any | Dict):
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        text = json.dumps(o, indent=4)
        await f.write(text)


def __repl_func(match):
    content = match.group()
    if '"' in content:
        return content
    else:
        return " ".join(content.split())


async def pretty_json_dump(path: str, o: List[Dict] | Dict | Any):
    async with aiofiles.open(os.path.join(path), "w", encoding="utf8") as f:
        output = json.dumps(o, indent=4)
        await f.write(re.sub(r"(?<=\[)[^\[\]]+(?=])", __repl_func, output))
