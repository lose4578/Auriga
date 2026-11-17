// ---- Agent backend vocabulary (design 5.3). One alias map; transcript rows stay in the Codex item vocabulary. ----
export const AGENT_EVENT_ALIASES = Object.freeze({
  'agent.turn_started': 'codex.turn/started',
  'agent.turn_completed': 'codex.turn/completed',
  'agent.connection_lost': 'codex.connection_lost',
  'agent.history_recovered': 'codex.history_recovered',
});
export const canonicalEventType = (type) => AGENT_EVENT_ALIASES[type] || type;
// Replace the backend name in static markup: `[data-agent-name]` text, and title/aria-label attributes written with "Codex".
export function applyAgentNames(root, name) {
  for (const element of root.querySelectorAll('[data-agent-name]')) element.textContent = name;
  for (const element of root.querySelectorAll('[title],[aria-label]')) {
    for (const attribute of ['title', 'aria-label']) {
      const key = attribute === 'title' ? 'agentTitle' : 'agentAria';
      const template = element.dataset[key] ?? element.getAttribute(attribute);
      if (template == null || !template.includes('Codex')) continue;
      element.dataset[key] = template;
      element.setAttribute(attribute, template.replaceAll('Codex', name));
    }
  }
}
// Any native agent event (Codex app-server `codex.*` or other backends' `agent.*`).
export const isNativeAgentEvent = (type) => typeof type === 'string' && (type.startsWith('codex.') || type.startsWith('agent.'));
const BACKEND_NAMES = { codex: 'Codex', claude_code: 'Claude Code', opencode: 'OpenCode' };
// Backend kind from the run snapshot `agent_backend` (string or {kind}); absent on old runs -> 'codex'.
export function backendKind(run) {
  const value = run?.agent_backend;
  const kind = typeof value === 'string' ? value : value?.kind;
  return typeof kind === 'string' && kind ? kind : 'codex';
}
export const backendName = (run) => BACKEND_NAMES[backendKind(run)] || backendKind(run);
// Turn status: `agent_status` when the server provides it, else the legacy `codex_status`.
export const agentStatus = (run) => run?.agent_status ?? run?.codex_status;
// The TUI tab is hidden only when the snapshot explicitly reports `terminal: null` (old runs omit the key).
export const terminalAbsent = (run) => run?.terminal === null;

// Build navigation from recorded native items and explicit event correlations.
// Native item order is the persisted record order; event sequence positions
// Harness actions that have no native item. No wall-clock association is made.
const TOOL_TYPES = new Set(['dynamicToolCall', 'mcpToolCall', 'commandExecution', 'fileChange', 'webSearch', 'imageGeneration', 'collabAgentToolCall']);
const CONTEXT_TYPES = new Set(['agentMessage', 'reasoning']);
const IDENTITY_KEYS = ['run_id', 'episode_id', 'env_instance_id', 'env_generation'];
const isTool = (item) => TOOL_TYPES.has(item.type) || Boolean(item.tool || item.command);
const scope = (item) => item.thread_id != null && item.turn_id != null
  ? JSON.stringify([item.thread_id, item.turn_id]) : JSON.stringify(['unscoped', item.id]);
const firstSeq = (events) => {
  const seqs = events.map((event) => event.server_seq).filter(Number.isFinite);
  return seqs.length ? Math.min(...seqs) : null;
};
const actionEntries = (actions) => actions instanceof Map ? [...actions.entries()] : Object.entries(actions || {});
const normalizeAction = (action_id, value) => ({ action_id, ...(value?.action ? value : value?.request ? { action: value } : {}) });

/**
 * actionsMap values may be {request, receipt} or {action: {request, receipt}, result}.
 * Each phase exposes actions as [{action_id, action?, result?}]. Missing action
 * bodies stay unresolved so callers can hydrate them through /v1/actions/:id.
 * before/after are recorded agentMessage/reasoning items from the same thread
 * and turn, bounded by neighboring tool calls. Empty reasoning stays empty.
 */
export function buildPhases(eventsArray = [], nativeItemsArray = [], actionsMap = new Map(), agentName = 'Codex') {
  const items = nativeItemsArray.filter((item) => item && typeof item.id === 'string');
  const byId = new Map(items.map((item) => [item.id, item]));
  const events = eventsArray.filter(Boolean);
  const itemEvents = new Map(), actionEvents = new Map();
  const actionRecords = new Map(actionEntries(actionsMap).map(([id, value]) => [id, normalizeAction(id, value)]));
  const matchingItem = (event) => {
    const c = event.correlation || {};
    const item = byId.get(c.item_id) || byId.get(c.tool_call_id);
    if (!item || ['thread_id', 'turn_id'].some((key) => c[key] != null && item[key] != null && c[key] !== item[key])) return null;
    return item;
  };
  for (const event of events) {
    const item = matchingItem(event);
    if (item) {
      if (!itemEvents.has(item.id)) itemEvents.set(item.id, []);
      itemEvents.get(item.id).push(event);
    }
    const actionId = event.correlation?.action_id;
    if (actionId) {
      if (!actionEvents.has(actionId)) actionEvents.set(actionId, []);
      actionEvents.get(actionId).push(event);
      if (!actionRecords.has(actionId)) actionRecords.set(actionId, { action_id: actionId });
    }
  }
  const toolActions = new Map(), unassociated = [];
  // Hydration requests finish concurrently. Their completion order must never
  // reorder actions inside one tool call or replay a later clip before an earlier one.
  const orderedActions = [...actionRecords].sort(([a], [b]) =>
    (firstSeq(actionEvents.get(a) || []) ?? Infinity) - (firstSeq(actionEvents.get(b) || []) ?? Infinity));
  for (const [id, record] of orderedActions) {
    const related = actionEvents.get(id) || [];
    const owners = new Set(related.map(matchingItem).filter((item) => item && isTool(item)).map((item) => item.id));
    // Conflicting explicit ownership is retained as Harness evidence instead of
    // arbitrarily assigning one action to either of two tools.
    if (owners.size === 1) {
      const owner = [...owners][0];
      if (!toolActions.has(owner)) toolActions.set(owner, []);
      toolActions.get(owner).push(record);
    } else unassociated.push({ record, events: related });
  }

  const groups = new Map();
  for (const item of items) {
    const key = scope(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  const phases = [], positions = new Map(items.map((item, index) => [item.id, index]));
  for (const group of groups.values()) {
    const toolIndices = group.flatMap((item, index) => isTool(item) ? [index] : []);
    const latestPrompt = (index) => group.slice(0, index + 1).findLast((item) => item.type === 'userMessage') || null;
    const context = (start, end) => group.slice(start, end).filter((item) => CONTEXT_TYPES.has(item.type));
    for (let n = 0; n < toolIndices.length; n += 1) {
      const index = toolIndices[n], item = group[index], related = itemEvents.get(item.id) || [];
      phases.push({
        id: `item:${item.id}`, title: item.tool || item.command || item.type, kind: 'tool', item, items: [item],
        before: context((toolIndices[n - 1] ?? -1) + 1, index),
        after: context(index + 1, toolIndices[n + 1] ?? group.length), prompt: latestPrompt(index),
        events: related, actions: toolActions.get(item.id) || [], seq: firstSeq(related),
        thread_id: item.thread_id, turn_id: item.turn_id, position: positions.get(item.id),
      });
    }
    // Retain the closing native phase even when there is no following tool.
    const tailStart = toolIndices.length ? toolIndices.at(-1) + 1 : 0;
    const tail = context(tailStart, group.length);
    if (tail.length) {
      const item = tail[0], related = tail.flatMap((entry) => itemEvents.get(entry.id) || []);
      phases.push({
        id: `codex:${item.id}`, title: toolIndices.length ? `${agentName} 后续记录` : `${agentName} 记录`, kind: 'codex',
        item, items: tail, before: [], after: [], prompt: latestPrompt(group.length - 1),
        events: related, actions: [], seq: firstSeq(related), thread_id: item.thread_id, turn_id: item.turn_id,
        position: positions.get(item.id),
      });
    } else if (!toolIndices.length) {
      const item = latestPrompt(group.length - 1);
      if (item) {
        const related = itemEvents.get(item.id) || [];
        phases.push({ id: `prompt:${item.id}`, title: '任务输入', kind: 'prompt', item, items: [item], before: [], after: [], prompt: item,
          events: related, actions: [], seq: firstSeq(related), thread_id: item.thread_id, turn_id: item.turn_id,
          position: positions.get(item.id) });
      }
    }
  }
  phases.sort((a, b) => a.position - b.position);
  const harnessPhases = unassociated.map(({ record, events: related }) => ({
    id: `action:${record.action_id}`, title: `Harness · ${record.action_id}`, kind: 'harness', item: null, items: [],
    before: [], after: [], prompt: null, events: related, actions: [record], seq: firstSeq(related),
    thread_id: related[0]?.correlation?.thread_id, turn_id: related[0]?.correlation?.turn_id,
  })).sort((a, b) => (a.seq ?? Infinity) - (b.seq ?? Infinity));
  for (const phase of harnessPhases) {
    const index = phase.seq === null ? -1 : phases.findIndex((other) => other.seq !== null && other.seq > phase.seq);
    if (index < 0) phases.push(phase); else phases.splice(index, 0, phase);
  }
  return phases.map(({ position, ...phase }) => phase);
}

/** Map only recorded exact boundaries. PTS is never derived from fps or time. */
export function mapActionVideo(record, action, result) {
  const request = action?.request, interval = action?.receipt?.execution_interval;
  const identity = record?.identity || { ...record?.event, ...record?.event?.payload };
  if (!request || IDENTITY_KEYS.some((key) => identity[key] === undefined || identity[key] !== request[key])) {
    return { reason: '动作与录像的运行 / episode / generation 不同或身份缺失' };
  }
  const entry = record?.entry, mapping = entry?.frame_mapping;
  if (!interval || !Number.isFinite(interval.start_sim_step) || !Number.isFinite(interval.end_sim_step)
      || interval.end_sim_step < interval.start_sim_step || !Array.isArray(mapping) || !mapping.length
      || mapping.some((frame) => !frame || !Number.isFinite(frame.sim_step) || !Number.isFinite(frame.pts_s) || frame.pts_s < 0)) {
    return { reason: '执行区间或逐帧 PTS 映射缺失 / 无效' };
  }
  if (mapping.some((frame, index) => index > 0 && (frame.pts_s < mapping[index - 1].pts_s || frame.sim_step < mapping[index - 1].sim_step))) {
    return { reason: '录像帧映射顺序不一致，无法确定执行区间' };
  }
  const camera = result?.cameras?.find((value) => value.camera_id === entry.camera_id);
  const anchor = camera?.frame_id && mapping.find((frame) => {
    const sourceDirectory = (frame.source_rgb || frame.source_png)?.split('/').at(-2);
    return frame.sim_step === result?.sim_step && (frame.frame_id === camera.frame_id
      || (sourceDirectory && `${sourceDirectory}-${entry.camera_id}` === camera.frame_id));
  });
  const frameDomains = [...new Set(mapping.map((frame) => frame.step_domain).filter(Boolean))];
  const declaredDomain = entry.step_domain || record.event?.step_domain
    || (mapping.every((frame) => frame.step_domain === mapping[0].step_domain) && mapping[0].step_domain);
  const domain = declaredDomain || (anchor && result.step_domain);
  if (!domain || domain !== interval.step_domain || frameDomains.some((value) => value !== domain)
      || (record.event?.clock_domain && result?.clock_domain !== record.event.clock_domain)) {
    return { reason: '录像步域 / 时钟未核对，需要声明或精确终态帧锚点' };
  }
  if (result && IDENTITY_KEYS.some((key) => result[key] !== undefined && result[key] !== request[key])) {
    return { reason: '执行结果观察与动作的运行 / episode / generation 不同' };
  }
  const start = mapping.find((frame) => frame.sim_step === interval.start_sim_step);
  const end = mapping.findLast((frame) => frame.sim_step === interval.end_sim_step);
  if (!start || !end || end.pts_s < start.pts_s) {
    return { reason: '录像未记录动作两端的精确映射，无法定位执行区间' };
  }
  return { start, end, domain, basis: declaredDomain ? 'declared_step_domain' : 'result_frame_anchor',
    pause: interval.start_sim_step === interval.end_sim_step || start.pts_s === end.pts_s };
}

/** Classify recorded boundaries without inferring frame times or sim time. */
export function clipTransition(previous, next) {
  const identity = (clip) => clip?.record?.identity || { ...clip?.record?.event, ...clip?.record?.event?.payload };
  const before = identity(previous), after = identity(next);
  const sameRecording = previous?.url && previous.url === next?.url
    && previous.record?.entry?.camera_id === next.record?.entry?.camera_id
    && previous.domain && previous.domain === next.domain
    && IDENTITY_KEYS.every((key) => before[key] != null && before[key] === after[key]);
  if (!sameRecording) return { kind: 'source', continuous: false };
  const end = previous.end, start = next.start;
  if (![end?.pts_s, start?.pts_s, next.end?.pts_s, end?.sim_step, start?.sim_step].every(Number.isFinite)) {
    return { kind: 'discontinuous', continuous: false };
  }
  const gap = start.pts_s - end.pts_s;
  if (end.sim_step === start.sim_step && gap <= 0 && next.end.pts_s > end.pts_s) {
    // Duplicate observations of the shared boundary may overlap in PTS. They
    // already played at the end of the previous clip; never replay those frames.
    return { kind: gap === 0 ? 'continuous' : 'overlap', continuous: true };
  }
  if (gap > 0) return { kind: 'gap', continuous: false, from: end.pts_s, to: start.pts_s };
  return { kind: 'discontinuous', continuous: false };
}
