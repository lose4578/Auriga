"""Registry lint: schema bounds, enums and defaults must also be stated in the description prose.

Why: OpenCode on the OpenAI Responses wire strips JSON Schema keywords (minimum, maximum, exclusiveMinimum,
exclusiveMaximum, default, minItems, maxItems, minLength, maxLength, ...) from MCP tool schemas before they reach the
model (spike 06 S6), and AGENTS.md requires every parameter's meaning, unit, default and range to live in the contract
text. The catalog is built exactly as an actor sees it (Workbench.actor_tools fallback runtime, pose_skill on, both
optional plugins registered) and every published tool's input schema is walked recursively.

Text visible to the model for a property = the property's OWN `description` plus the TOOL's description. (Ancestor
property descriptions are deliberately not counted: a stripped-schema client may flatten nested objects.)

Matching rules (conservative; a property is checked once per keyword it carries):
* numeric keywords (minimum, maximum, exclusiveMinimum, exclusiveMaximum, minItems, maxItems, minLength, maxLength):
  the text must contain a numeric token whose value equals the bound. Tokens are decimal or scientific literals
  (`0.05`, `-1`, `1e-3`, `1.5E+2`) and thousands-free integers; unit suffixes do not matter (`5 mm` still yields 5),
  but unit conversion is NOT applied (0.005 m is not matched by "5 mm"). Non-negative integers up to 12 may instead
  be written as an English word delimited by non-letters (`Exactly three finite numbers` states minItems=maxItems=3).
* enum: every enum value must occur in the text, case-sensitively, delimited by non-identifier characters
  (so `a` does not match inside `data`). Non-string values are matched through their JSON rendering (`true`, `3`).
* default: the text must contain the word `default` (case-insensitive, also `defaults`/`defaulting`), and, when the
  default is a number or a short string, that value must also be present as a token / delimited word. Container
  defaults (empty or not) only require the word `default`.
"""
import json
import math
import re
from types import SimpleNamespace
import unittest

from auriga.core.config import Config
from test_tool_parallel_contract import full_registry

NUMERIC_KEYWORDS = ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
                    'minItems', 'maxItems', 'minLength', 'maxLength')
NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def actor_catalog():
    config = Config.from_dict({'features': {'pose_skill': True}})
    runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
        observation_profile=config.features.observation_profile, motion_review=config.features.motion_review,
        enable_wait=True, vision_compare=config.features.vision_compare, visual_reviewer=object(),
        action_policy=True, capabilities=None, programs=True)
    return full_registry().catalog(runtime, role='actor')


def numbers_in(text):
    values = []
    for token in NUMBER.findall(text):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                'twelve')


def has_number(text, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if any(math.isclose(found, float(value), rel_tol=1e-9, abs_tol=1e-12) for found in numbers_in(text)):
        return True
    return float(value) in range(len(NUMBER_WORDS)) and re.search(
        r'(?<![A-Za-z])' + NUMBER_WORDS[int(value)] + r'(?![A-Za-z])', text, re.I) is not None


def has_word(text, word):
    return re.search(r'(?<![\w])' + re.escape(word) + r'(?![\w])', text) is not None


def has_value(text, value):
    if isinstance(value, bool) or value is None:
        return has_word(text, json.dumps(value))
    if isinstance(value, (int, float)):
        return has_number(text, value)
    return has_word(text, str(value))


def walk(schema, path, root):
    """Yield (json_path, subschema) for every schema node below and including `schema`."""
    if not isinstance(schema, dict):
        return
    ref = schema.get('$ref')
    if ref and ref.startswith('#/'):
        target = root
        for part in ref[2:].split('/'):
            target = target[part]
        yield from walk(target, path, root)
    yield path, schema
    for key, child in (schema.get('properties') or {}).items():
        yield from walk(child, path + '/properties/' + key, root)
    for key in ('items', 'additionalProperties', 'not', 'if', 'then', 'else'):
        if isinstance(schema.get(key), dict):
            yield from walk(schema[key], path + '/' + key, root)
    for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
        for index, child in enumerate(schema.get(key) or []):
            yield from walk(child, path + '/%s/%d' % (key, index), root)
    for key in ('$defs', 'definitions'):
        for name, child in (schema.get(key) or {}).items():
            yield from walk(child, path + '/%s/%s' % (key, name), root)


def gaps(tool):
    """Return (path, keyword, detail) for every schema constraint missing from the visible prose."""
    found = []
    schema = tool['inputSchema']
    for path, node in walk(schema, '', schema):
        text = (node.get('description') or '') + '\n' + (tool.get('description') or '')
        for keyword in NUMERIC_KEYWORDS:
            if keyword in node and not has_number(text, node[keyword]):
                found.append((path, keyword, node[keyword]))
        if 'enum' in node:
            for value in node['enum']:
                if not has_value(text, value):
                    found.append((path, 'enum', value))
        if 'default' in node:
            value = node['default']
            scalar = isinstance(value, (bool, int, float)) or value is None or (isinstance(value, str) and value)
            if not re.search(r'defaults?\b|defaulting', text, re.I) or (scalar and not has_value(text, value)):
                found.append((path, 'default', value))
    return found


class MatcherTest(unittest.TestCase):
    def test_numbers(self):
        self.assertTrue(has_number('range 0.05 to 1.5 m', 0.05))
        self.assertTrue(has_number('at most 1e-3 rad', 0.001))
        self.assertTrue(has_number('between -1 and 1', -1))
        self.assertTrue(has_number('limit 12 items', 12))
        self.assertFalse(has_number('limit 120 items', 12))
        self.assertFalse(has_number('about 5 mm', 0.005))
        self.assertFalse(has_number('v1.25', 1.2))
        self.assertTrue(has_number('Exactly three finite numbers', 3))
        self.assertFalse(has_number('Exactly three finite numbers', 2))
        self.assertFalse(has_number('someone', 1))

    def test_words(self):
        self.assertTrue(has_word('mode a or b', 'a'))
        self.assertFalse(has_word('some data', 'a'))
        self.assertTrue(has_value('flag is true', True))
        self.assertTrue(has_value('pick contact_stop', 'contact_stop'))

    def test_gap_detection_and_walk(self):
        tool = dict(name='t', description='Tool. Up to 5 things.', inputSchema=dict(type='object', properties=dict(
            a=dict(type='number', minimum=0.1, description='x'),
            b=dict(type='array', maxItems=5, items=dict(type='string', enum=['p', 'q'], description='p only'))),
            required=[]))
        self.assertEqual([(p, k) for p, k, _ in gaps(tool)],
                         [('/properties/a', 'minimum'), ('/properties/b/items', 'enum')])


class RegistryProseBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = actor_catalog()

    def test_catalog_contains_both_plugins_and_pose_skill(self):
        names = {tool['name'] for tool in self.catalog}
        self.assertTrue({'robotic_view', 'stereo_guide', 'pose_plan', 'robot_action', 'program_run'} <= names)

    def test_every_bound_enum_and_default_is_stated_in_prose(self):
        problems = []
        for tool in self.catalog:
            problems += ['%s%s %s=%r' % (tool['name'], path, keyword, value) for path, keyword, value in gaps(tool)]
        self.assertEqual(problems, [], '\n' + '\n'.join(problems))

    def test_top_level_combinator_rules_are_stated_in_prose(self):
        # MCP publishes inputSchema without top-level anyOf/oneOf/allOf/if/then/else (mcp_server.portable_input_schema;
        # Claude Code drops such tools). The generated "input: ... Object rules: ..." clause must carry them.
        from auriga.cortex.agents.mcp_server import TOP_LEVEL_COMBINATORS
        phrases = {'anyOf': 'at least one of (', 'oneOf': 'exactly one of (', 'allOf': 'all of (', 'if': 'if ('}
        problems = []
        for tool in self.catalog:
            present = [key for key in TOP_LEVEL_COMBINATORS if key in tool['inputSchema']]
            guide = [line for line in tool['description'].splitlines() if line.startswith('input: ')]
            for key in present:
                if key in phrases and not any('Object rules:' in line and phrases[key] in line for line in guide):
                    problems.append('%s top-level %s is not stated in the description' % (tool['name'], key))
            if any(key in ('then', 'else') for key in present) and not any('if (' in line for line in guide):
                problems.append('%s top-level then/else is not stated in the description' % tool['name'])
        self.assertEqual(problems, [])


if __name__ == '__main__':
    unittest.main()
