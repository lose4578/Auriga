import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { AGENT_EVENT_ALIASES, canonicalEventType, isNativeAgentEvent, backendKind, backendName, agentStatus,
  terminalAbsent, applyAgentNames, buildPhases } from './phase-model.js';

const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/agent-backend-claude-code.json', import.meta.url), 'utf8'));

test('one alias map ties agent.* events to the codex.* vocabulary', () => {
  assert.deepEqual(AGENT_EVENT_ALIASES, {
    'agent.turn_started': 'codex.turn/started', 'agent.turn_completed': 'codex.turn/completed',
    'agent.connection_lost': 'codex.connection_lost', 'agent.history_recovered': 'codex.history_recovered' });
  assert.equal(canonicalEventType('agent.turn_started'), 'codex.turn/started');
  assert.equal(canonicalEventType('codex.turn/started'), 'codex.turn/started');
  assert.equal(canonicalEventType('action.completed'), 'action.completed');
  assert.equal(canonicalEventType('agent.native'), 'agent.native');
  assert.ok(isNativeAgentEvent('agent.native') && isNativeAgentEvent('codex.error') && !isNativeAgentEvent('action.started'));
  assert.ok(!isNativeAgentEvent(undefined));
});

test('backend kind defaults to Codex and maps display names', () => {
  assert.equal(backendName({}), 'Codex');
  assert.equal(backendName(undefined), 'Codex');
  assert.equal(backendName({ agent_backend: 'codex' }), 'Codex');
  assert.equal(backendName({ agent_backend: 'claude_code' }), 'Claude Code');
  assert.equal(backendName({ agent_backend: { kind: 'opencode' } }), 'OpenCode');
  assert.equal(backendName({ agent_backend: 'future_agent' }), 'future_agent');
  assert.equal(backendKind({ agent_backend: null }), 'codex');
});

test('agent_status wins over codex_status, old runs fall back', () => {
  assert.equal(agentStatus({ codex_status: 'running' }), 'running');
  assert.equal(agentStatus({ agent_status: 'completed', codex_status: 'running' }), 'completed');
  assert.equal(agentStatus({}), undefined);
});

test('TUI tab is hidden only for an explicit null terminal', () => {
  assert.equal(terminalAbsent({ terminal: null }), true);
  assert.equal(terminalAbsent({}), false);
  assert.equal(terminalAbsent({ terminal: { exit_code: null } }), false);
  assert.equal(terminalAbsent(fixture.run), true);
  assert.equal(terminalAbsent(fixture.legacy_codex_run), false);
});

test('applyAgentNames rewrites marked text and Codex-bearing attributes, reversibly', () => {
  const attrs = new Map([['title', '中断 Codex 轮次']]);
  const span = { textContent: 'Codex' };
  const button = { dataset: {}, getAttribute: (k) => attrs.get(k) ?? null, setAttribute: (k, v) => attrs.set(k, v) };
  const root = { querySelectorAll: (selector) => selector === '[data-agent-name]' ? [span] : [button] };
  applyAgentNames(root, 'Claude Code');
  assert.equal(span.textContent, 'Claude Code');
  assert.equal(attrs.get('title'), '中断 Claude Code 轮次');
  applyAgentNames(root, 'Codex');
  assert.equal(attrs.get('title'), '中断 Codex 轮次');
});

test('phases of a non-Codex run render with agent.* events and a backend-named closing phase', () => {
  const phases = buildPhases(fixture.events, fixture.items, new Map(), backendName(fixture.run));
  const tool = phases.find((phase) => phase.item?.id === 'tool-1');
  assert.ok(tool, 'tool phase exists');
  assert.equal(tool.prompt.id, 'prompt');
  assert.deepEqual(tool.before.map((item) => item.id), ['plan']);
  const tail = phases.find((phase) => phase.kind === 'codex');
  assert.equal(tail.title, 'Claude Code 后续记录');
});

test('old Codex runs build exactly the same phases as before (default name)', () => {
  const codexEvents = fixture.events.map((event) => ({ ...event, type: canonicalEventType(event.type) }));
  const alias = buildPhases(fixture.events, fixture.items);
  const legacy = buildPhases(codexEvents, fixture.items);
  assert.equal(alias.find((phase) => phase.kind === 'codex').title, 'Codex 后续记录');
  assert.deepEqual(alias.map((phase) => [phase.id, phase.kind, phase.title, phase.items?.map((i) => i.id)]),
    legacy.map((phase) => [phase.id, phase.kind, phase.title, phase.items?.map((i) => i.id)]));
});
