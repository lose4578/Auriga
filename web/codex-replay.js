const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
function savedText(item) {
  const parts = [item?.text, ...(Array.isArray(item?.summary) ? item.summary : []), ...(Array.isArray(item?.content) ? item.content : [])];
  return [...new Set(parts.map((part) => typeof part === 'string' ? part : part?.text || '').filter(Boolean))].join('\n');
}
function details(title, value, open = false) {
  const element = node('details'); element.open = open;
  element.append(node('summary', title), node('pre', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
  return element;
}
export function createCodexReplay({ resizeTerminal, agentName = () => 'Codex' }) {
  const $ = (id) => document.getElementById(id), content = $('codex-replay-content');
  let readError;
  let signature, selectedPhase, position = { index: -1, total: 0 }, nativeCount = 0;
  function setTab(tab) {
    const raw = tab === 'terminal';
    $('codex-replay').hidden = raw; $('raw-terminal').hidden = !raw;
    for (const name of ['records', 'terminal']) $(`codex-tab-${name}`).setAttribute('aria-selected', String(name === tab));
    if (raw) requestAnimationFrame(resizeTerminal);
  }
  $('codex-tab-records').onclick = () => { setTab('records'); };
  $('codex-tab-terminal').onclick = () => { setTab('terminal'); };
  function appendItem(parent, item, label) {
    if (!item) return;
    const article = node('article', undefined, `codex-record ${item.type === 'reasoning' ? 'codex-reasoning' : ''}`);
    article.dataset.itemId = item.id || '';
    const title = label || ({ userMessage: '用户指令', agentMessage: `${agentName()} 说明`, reasoning: '已保存的思考摘要' }[item.type]) || item.tool || item.command || item.type;
    article.append(node('h3', title));
    const text = savedText(item);
    if (text) article.append(node('pre', text, 'codex-message'));
    else if (item.type === 'reasoning') article.append(node('p', '此条 reasoning 记录没有可见文本。', 'codex-record-note'));
    const args = item.arguments;
    if (args !== undefined) {
      const fields = args && typeof args === 'object' ? ['language', 'prompt', 'instruction', 'code', 'script'].filter((key) => args[key] !== undefined) : [];
      for (const key of fields) article.append(details(`工具输入 · ${key}`, args[key], true));
      article.append(details('完整工具参数', args, !fields.length));
    }
    if (item.command) article.append(details('执行命令', item.command, true));
    for (const [key, title] of [['contentItems', '工具返回'], ['result', '工具结果'], ['aggregatedOutput', '执行输出'], ['error', '错误']]) {
      if (item[key] !== undefined && item[key] !== null) article.append(details(title, item[key]));
    }
    if (item.status || typeof item.success === 'boolean') article.append(node('p', `工具记录状态：${item.status || '未提供'}${typeof item.success === 'boolean' ? ` · 工具调用 ${item.success ? '成功' : '失败'}` : ''}`, 'codex-record-note'));
    article.append(details(`原始 ${agentName()} 记录`, item)); parent.append(article);
  }
  function render() {
    const phase = selectedPhase;
    const nextSignature = JSON.stringify([phase?.id, phase?.items, phase?.before, phase?.after, phase?.prompt, phase?.actions, position, nativeCount, readError, agentName()]);
    if (signature === nextSignature) return; const changed = content.dataset.phaseId !== phase?.id; signature = nextSignature;
    content.replaceChildren(); content.dataset.phaseId = phase?.id || '';
    $('codex-replay-status').textContent = readError ? `${agentName()} 记录读取失败 · ${readError}` : phase ? `阶段 ${position.index + 1} / ${position.total} · ${phase.title}` : nativeCount ? '正在按工具和动作整理阶段' : `等待已保存的 ${agentName()} 记录`;
    if (!phase) { content.append(node('p', '尚无可显示的阶段记录。', 'codex-record-note')); return; }
    if (phase.prompt) content.append(details('本轮用户指令', savedText(phase.prompt) || phase.prompt));
    const appendGroup = (title, items) => {
      if (!items?.length) return;
      content.append(node('h4', title, 'codex-group-title'));
      for (const item of items) appendItem(content, item);
    };
    appendGroup('执行前', phase.before);
    appendGroup('当前阶段', phase.items);
    if (!phase.items?.length) content.append(node('p', `此阶段为 Harness 动作，没有关联的 ${agentName()} 消息。`, 'codex-record-note'));
    for (const entry of phase.actions || []) {
      const receipt = entry.action?.receipt;
      if (receipt) content.append(details(`机器人动作回执 · ${receipt.status || '未知'}`, { action_id: entry.action_id, ...receipt }));
    }
    appendGroup('执行后', phase.after);
    if (changed) content.scrollTop = 0;
  }
  return {
    setPhase(phase, value) { selectedPhase = phase; position = value; render(); },
    setItems(items) { nativeCount = items.length; readError = undefined; render(); },
    unavailable(message) { readError = message; render(); },
    terminalFailure(message) { $('codex-terminal-notice').hidden = false; $('codex-terminal-notice').textContent = message; },
    clearTerminalFailure() { $('codex-terminal-notice').hidden = true; $('codex-terminal-notice').textContent = ''; },
    setTab,
  };
}
