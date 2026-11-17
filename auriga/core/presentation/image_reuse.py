"""Exact image reuse within one model request, retaining each original."""
from copy import deepcopy
import json

IMAGE_REUSE_POLICY = "exact_image_part_current_request_v1"


def responses_image_slots(request):
    """Default part locator: image parts of a Responses request, in context order.

    Yields ``(parts, index, inline_data)`` for every ``input_image`` part inside an ``input``
    message's ``content`` list or a ``function_call_output`` / ``custom_tool_call_output``
    ``output`` list. ``inline_data`` is the ``image_url`` string when it is a
    ``data:image/`` URL, else None (a displayed image that is never deduplicated). Yields
    nothing when ``input`` is not a list. Generated lazily, so the caller may replace the
    part at ``parts[index]`` before the next slot is produced.
    """
    items = request.get("input")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type", "message") == "message":
            key = "content"
        elif item.get("type") in ("function_call_output", "custom_tool_call_output"):
            key = "output"
        else:
            continue
        parts = item.get(key)
        if not isinstance(parts, list):
            continue
        for index, part in enumerate(parts):
            if not isinstance(part, dict) or part.get("type") != "input_image":
                continue
            url = part.get("image_url")
            yield parts, index, url if isinstance(url, str) and url.startswith("data:image/") else None


def responses_text_part(text):
    """Default reference part: a Responses ``input_text`` part carrying ``text``."""
    return dict(type="input_text", text=text)


def reuse_exact_images(request, *, locate=None, text_part=None):
    """Keep original image bytes, replacing later equal blocks with references.

Only typed image parts in model input message/tool-output arrays are candidates.
Identity covers the complete image part (including detail), not approximate
pixels. The map is local to this request: compaction/resume cannot create a
reference to an image omitted from the current context. No input is mutated.

locate: part locator, ``locate(request_copy)`` yielding ``(parts, index, inline_data)``
for every image part in context order (``inline_data`` = inline payload string counted in
``saved_url_chars``, or None for an image that is displayed but never deduplicated).
Default ``responses_image_slots`` (OpenAI Responses shape). text_part: ``text_part(text)``
returns the part that replaces a duplicate; default ``responses_text_part``.
Returns ``(request_copy, {images, replaced, saved_url_chars})``.
    """
    locate = responses_image_slots if locate is None else locate
    text_part = responses_text_part if text_part is None else text_part
    result = deepcopy(request)
    seen = {}
    images = replaced = saved_url_chars = displayed = 0
    for parts, index, data in locate(result):
        if data is None:
            displayed += 1
            continue
        images += 1
        identity = json.dumps(parts[index], sort_keys=True, separators=(",", ":"))
        if identity not in seen:
            displayed += 1
            seen[identity] = displayed
            continue
        number = seen[identity]
        parts[index] = text_part(
            f"[Exact duplicate image: identical pixels to image {number} "
            "earlier in this context, counting actual images only. "
            "Use that image for this observation; no visual change.]")
        replaced += 1
        saved_url_chars += len(data)
    return result, dict(images=images, replaced=replaced, saved_url_chars=saved_url_chars)
