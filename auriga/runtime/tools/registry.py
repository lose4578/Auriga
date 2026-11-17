"""One schema, availability and role check for catalog and actual invocation."""
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
import re
from typing import Callable

from jsonschema import Draft202012Validator, validators

from ...core.contracts import canonical, digest
from ...core.tool_errors import ToolInputError, schema_input_error
from ...core.tool_output import (InvalidContentItem, ToolResponse, output_contract_error,
                           validate_content_part, validate_display)
from .context import ToolAvailability, ToolContext


def _number(checker, value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


Validator = validators.extend(Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine('number', _number))


@lru_cache(maxsize=256)
def _checked_schema(schema_json):
    # The immutable key includes the complete schema; edits cannot reuse an
    # old validator, and separate runs share no mutable declaration object.
    schema = json.loads(schema_json)
    Validator.check_schema(schema)
    return schema


def validator_for(schema):
    # Callers may inspect/edit validator.schema without changing a live contract.
    return Validator(deepcopy(_checked_schema(canonical(schema))))


FUNCTION_SPEC_SCHEMA = dict(type='object', required=['type', 'name', 'description', 'inputSchema'],
    properties=dict(type=dict(const='function'), name=dict(type='string', minLength=1, maxLength=128,
        pattern=r'^[A-Za-z0-9_-]+$'),
        description=dict(type='string'), inputSchema={}, deferLoading=dict(type='boolean')))


def _copy_spec(spec):
    return replace(spec, definition=deepcopy(spec.definition), roles=frozenset(spec.roles),
        legacy_schema=deepcopy(spec.legacy_schema), output_schemas=deepcopy(spec.output_schemas))


def _check_examples(schema, reason):
    validator = validator_for(schema)
    for example in schema.get('examples', []) if isinstance(schema, dict) else ():
        validate_display(example)
        if not validator.is_valid(example):
            raise ValueError(reason)


@dataclass(frozen=True)
class ToolSpec:
    definition: dict
    handler: Callable
    roles: frozenset = frozenset({'actor', 'program', 'trusted_legacy'})
    effect: str | Callable = 'read'
    unavailable: Callable | None = None
    legacy_schema: dict | None = None
    api_version: str = "1.0"
    # Each discriminator owns a JSON Schema for ToolResponse.data. A mapping
    # keeps multi-result tools explicit without coupling them to request modes.
    output_schemas: dict | None = None
    # Concurrency contract for agent tool transports: True means a call is safe
    # to run concurrently with other parallel-safe calls because it mutates no
    # Harness/world state (e.g. no fresh capture, plan/memory write or motion).
    # A callable receives the call's argument object and returns that verdict per
    # request. Default False: the call runs in arrival order behind every
    # earlier call. This is scheduling only; it never changes what a tool does.
    parallel: bool | Callable[[dict], bool] = False

    @property
    def name(self):
        return self.definition['name']

    def effect_for(self, arguments):
        return self.effect(arguments) if callable(self.effect) else self.effect

    def parallel_for(self, arguments):
        """Return whether this call may overlap other parallel-safe calls."""
        return bool(self.parallel(arguments)) if callable(self.parallel) else self.parallel


class ToolRegistry:
    def __init__(self, specs=()):
        self.__specs = {}
        for spec in specs:
            self.register(spec)

    @property
    def _specs(self):
        """Compatibility inspection returns isolated contracts, never live storage."""
        return {name: _copy_spec(spec) for name, spec in self.__specs.items()}

    def register(self, spec):
        if not isinstance(spec, ToolSpec):
            raise ValueError('invalid_tool_spec')
        validate_display(spec.definition)
        issue = next(validator_for(FUNCTION_SPEC_SCHEMA).iter_errors(spec.definition), None)
        if issue is not None:
            raise schema_input_error(issue, code='invalid_tool_definition')
        if spec.name != spec.name.strip():
            raise ToolInputError('invalid_tool_definition', field_path='/name', actual=spec.name,
                suggestion='Native dynamic tool names cannot have leading or trailing whitespace.')
        if spec.name == 'mcp' or spec.name.startswith('mcp__'):
            raise ToolInputError('reserved_tool_name', field_path='/name', actual=spec.name,
                suggestion='Native Codex reserves mcp and the mcp__ name prefix; choose another function name.')
        for field in ('namespace', 'exposeToContext'):
            if field in spec.definition:
                raise ToolInputError('invalid_tool_definition', field_path='/' + field,
                    actual=spec.definition[field], suggestion='Do not mix legacy dynamic-tool fields with a canonical function declaration.')
        spec = _copy_spec(spec)
        if not isinstance(spec.api_version, str) or re.fullmatch(r'[1-9][0-9]*\.[0-9]+', spec.api_version) is None:
            raise ValueError('invalid_tool_api_version')
        if type(spec.parallel) is not bool and not callable(spec.parallel):
            raise ValueError('invalid_tool_parallel:' + spec.name)
        if spec.name.startswith('scene_') and spec.roles & {'actor', 'program'}:
            raise ValueError('reserved_tool_namespace:scene_')
        if spec.name in self.__specs:
            raise ValueError('duplicate_tool:' + spec.name)
        _check_examples(spec.definition['inputSchema'], 'invalid_tool_input_example:' + spec.name)
        schema = spec.definition['inputSchema']
        if isinstance(schema, dict) and schema.get('type') in ('null', ['null']):
            raise ToolInputError('unsupported_tool_input_schema', field_path='/inputSchema/type', actual=schema['type'],
                suggestion='Native Codex rejects a singleton null tool input type; declare the actual input structure.')
        if spec.legacy_schema is not None:
            validator_for(spec.legacy_schema)
        if spec.output_schemas is None:
            if spec.roles != frozenset({'trusted_legacy'}):
                raise ValueError('tool_output_contract_required:' + spec.name)
        elif not isinstance(spec.output_schemas, dict) or not spec.output_schemas:
            raise ValueError('invalid_tool_output_contract:' + spec.name)
        else:
            for kind, schema in spec.output_schemas.items():
                if not isinstance(kind, str) or not kind or not isinstance(schema, dict):
                    raise ValueError('invalid_tool_output_contract:' + spec.name)
                _check_examples(schema, 'invalid_tool_output_example:' + spec.name + ':' + kind)
        self.__specs[spec.name] = spec

    @property
    def names(self):
        return frozenset(self.__specs)

    def parallel_safe(self, name, arguments):
        """Classify one agent tool call for concurrent dispatch.

        True only when ``name`` is a registered tool whose ``ToolSpec.parallel``
        contract allows this ``arguments`` object to run concurrently with other
        parallel-safe calls. An unknown tool, a non-object argument value or a
        failing request-dependent classifier yields False, so the call is ordered
        (the safe direction). Role and availability are still enforced by
        ``validate``/``invoke`` when the call executes.
        """
        spec = self.__specs.get(name) if isinstance(name, str) else None
        if spec is None or not isinstance(arguments, dict):
            return False
        try:
            return spec.parallel_for(arguments) is True
        except Exception:
            return False

    def spec(self, name, *, role):
        spec = self.__specs.get(name) if isinstance(name, str) else None
        if spec is None:
            raise ToolInputError('unknown_tool', field_path='/tool', actual=name,
                suggestion='Use a tool in the current published catalog. Registered names for this role (some may be unavailable in this run): ' +
                    ', '.join(sorted(spec.name for spec in self.__specs.values() if role in spec.roles)) + '.')
        if role not in spec.roles:
            raise ToolInputError('tool_not_permitted_for_role', field_path='/tool', actual=name,
                suggestion='This tool is not exposed to your role. Use the current published tool catalog; do not call trusted legacy or evaluator interfaces.')
        return _copy_spec(spec)

    def unavailable(self, spec, runtime, arguments):
        if spec.name in runtime.disabled_tools:
            return 'tool_disabled_by_config:' + spec.name
        return spec.unavailable(ToolAvailability.from_runtime(runtime), arguments) if spec.unavailable else None

    def validate(self, name, arguments, *, role, runtime=None):
        spec = self.spec(name, role=role)
        if runtime is not None:
            reason = self.unavailable(spec, runtime, {})
            if reason:
                raise self._unavailable_error(reason, name, runtime, role)
        schema = spec.legacy_schema if role == 'trusted_legacy' and spec.legacy_schema else spec.definition['inputSchema']
        issue = next(validator_for(schema).iter_errors(arguments), None)
        if issue is not None:
            raise schema_input_error(issue)
        if runtime is not None:
            reason = self.unavailable(spec, runtime, arguments)
            if reason:
                raise self._unavailable_error(reason, name, runtime, role)
        return spec

    def _unavailable_error(self, reason, name, runtime, role):
        choices = [item['name'] for item in self.catalog(runtime, role=role)]
        return ToolInputError(reason, field_path='/tool', actual=name,
            suggestion='Unavailable under the current run configuration/capabilities: ' + reason +
                '. Changing argument spelling cannot enable this capability. Available tools: ' + ', '.join(choices) +
                '. For a rejected optional mode, use only a supported mode with evidence actually available; otherwise report the missing capability. Do not fabricate evidence, bypass the run policy, or retry an uncertain physical action.')

    def invoke(self, name, arguments, correlation, *, runtime, role):
        spec = self.validate(name, arguments, role=role, runtime=runtime)
        context = ToolContext.from_runtime(runtime, name, spec.effect_for(arguments), correlation)
        response = spec.handler(context, arguments, correlation)
        if spec.output_schemas is None:  # Explicit trusted-legacy wire boundary.
            return response
        if not isinstance(response, ToolResponse):
            return output_contract_error(response, name, 'typed_result_required', '/kind')
        if type(response.success) is not bool:
            return output_contract_error(response, name, 'result_success_not_boolean', '/success')
        if not isinstance(response.kind, str) or response.kind not in spec.output_schemas:
            return output_contract_error(response, name, 'unexpected_result_kind', '/kind')
        try:
            validate_display(response.data)
        except (TypeError, ValueError, OverflowError):
            return output_contract_error(response, name, 'result_data_not_json', '/data')
        issue = next(validator_for(spec.output_schemas[response.kind]).iter_errors(response.data), None)
        if issue is not None:
            diagnostic = schema_input_error(issue, code='invalid_tool_result', prefix=('data',))
            return output_contract_error(response, name, 'result_schema_mismatch', diagnostic.field_path)
        if not isinstance(response.content, list):
            return output_contract_error(response, name, 'display_parts_must_be_list', '/content')
        for index, part in enumerate(response.content):
            try:
                validate_content_part(part)
            except InvalidContentItem as error:
                return output_contract_error(response, name, 'invalid_native_content_item',
                    '/content/' + str(index) + error.field_path)
            except (TypeError, ValueError, OverflowError):
                return output_contract_error(response, name, 'display_not_json', '/content/' + str(index))
        return response

    def catalog(self, runtime, *, role='actor'):
        return [deepcopy(spec.definition) for spec in self.__specs.values()
                if role in spec.roles and self.unavailable(spec, runtime, {}) is None]

    def manifest(self, runtime, *, role='actor'):
        catalog = self.catalog(runtime, role=role)
        return dict(role=role, catalog_hash=digest(catalog), tools=[dict(
            name=spec.name, api_version=spec.api_version, schema_hash=digest(spec.definition['inputSchema']),
            result_kinds=list(spec.output_schemas) if spec.output_schemas is not None else None,
            output_schemas=deepcopy(spec.output_schemas),
            output_schema_hash=digest(spec.output_schemas) if spec.output_schemas is not None else None,
            output_contract='typed_data' if spec.output_schemas is not None else 'trusted_legacy_wire_unvalidated',
            effect=spec.effect if isinstance(spec.effect, str) else 'request_dependent',
            parallel=spec.parallel if isinstance(spec.parallel, bool) else 'request_dependent',
            available=(reason := self.unavailable(spec, runtime, {})) is None,
            unavailable_reason=reason) for spec in self.__specs.values() if role in spec.roles])
