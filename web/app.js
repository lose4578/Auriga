import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { createPhaseReview } from './phase-review.js';
import { mapActionVideo, canonicalEventType, isNativeAgentEvent, backendName, agentStatus, terminalAbsent, applyAgentNames } from './phase-model.js';
import { renderTaskReview } from './task-review.js';
import { createCodexReplay } from './codex-replay.js';
import { createRecordCache } from './record-cache.js';

const $ = (id) => document.getElementById(id);
const runPrefix = location.pathname.match(/^\/runs\/[^/]+/)?.[0] || '';
const apiURL = (path) => path?.startsWith('/v1/') ? `${runPrefix}${path}` : path;
$('records-events').href = apiURL('/v1/events?after=0');
const historyMode = () => currentRun && (currentRun.library?.live === false || currentRun.view_mode === 'replay' || currentRun.status === 'replay' || agentStatus(currentRun) === 'replay');
const readOnly = () => !currentRun || currentRun.read_only === true || currentRun.writable === false || historyMode();
if (runPrefix) { $('library-link').hidden = false; const saved = sessionStorage.getItem('auriga-library-url'); if (saved?.startsWith('/?')) $('library-link').href = saved; }
const terminal = new Terminal({
  disableStdin: true, convertEol: false, cursorBlink: false, scrollback: 5000,
  fontSize: 13, fontFamily: 'ui-monospace, "DejaVu Sans Mono", monospace',
  theme: { background: '#141b18', foreground: '#d2dfd6', cursor: '#91aa98', selectionBackground: '#34533e' },
});
const fit = new FitAddon();
terminal.loadAddon(fit);
terminal.open($('terminal'));
let terminalBootstrapText = '', terminalBootstrapKnown, terminalReplayInfo, terminalStreamError;
let socket, terminalCursor = 0, terminalReplayTarget = 0, terminalWrite = Promise.resolve(), terminalSession, terminalEnded = false;
let latestObservation, displayedObservation, replay = false, selectedEvent;
let eventCursor = 0, selectionVersion = 0;
const rows = new Map();
const seenEvents = new Set();
const videoRecords = new Map();
let selectedVideoAction;
const clientId = crypto.randomUUID();
let writerToken, currentRun, observations = [], nativeItems = [], pendingRequests = [], sceneVersions = [];
let refreshTimer, taskAttempt, stopAttempt;
const recordCaches = Object.fromEntries(['observations', 'items', 'requests'].map((name) => [name, createRecordCache()]));
let incrementalRecords = true, historicalWindow = false, historyBefore, eventsWindowed = false;
const gapMessages = new Set();
const agentName = () => backendName(currentRun);
const codexReplay = createCodexReplay({ resizeTerminal: sizeTerminal, agentName });
const phaseReview = createPhaseReview({ read: getJSON, artifactURL, terminal, inspectEvent, onPhaseChange: (phase, position) => codexReplay.setPhase(phase, position),
  terminalReady: async () => { await terminalWrite; return Boolean(terminalSession) && terminalCursor >= terminalReplayTarget; } });

async function getJSON(path, options) {
  if (options?.method && options.method !== 'GET' && readOnly()) throw new Error('只读回放不可修改运行');
  const response = await fetch(apiURL(path), options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).error || message; } catch { /* HTTP status remains available. */ }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function applyTerminalDiagnosis(status) {
  terminalReplayInfo = status?.replay || terminalReplayInfo;
  updateTerminalNote();
  if (typeof status?.bootstrap_failed !== 'boolean') return;
  terminalBootstrapKnown = status.bootstrap_failed;
  if (status.bootstrap_failed) codexReplay.terminalFailure(`原生终端初始化失败；请在阶段记录查看已保存的 ${agentName()} 消息与工具结果。`);
  else { terminalBootstrapText = ''; codexReplay.clearTerminalFailure(); }
}

function updateTerminalNote() {
  const recovered = terminalReplayInfo?.kind === 'recovered_native_tui';
  document.querySelector('.terminal-note').textContent = recovered
    ? `原生 ${agentName()} TUI · 从保存的会话归档恢复 · ${terminalReplayInfo.recovered_at ? new Date(terminalReplayInfo.recovered_at).toLocaleString('zh-CN', { hour12: false }) : ''} · 只读`
    : historyMode() ? '持久终端日志 · 只读回放' : '原生 TUI · 只读监看';
  $('terminal-original-log').hidden = !terminalReplayInfo?.original_bootstrap_failed;
  $('terminal-original-log').href = apiURL('/v1/terminal-log');
  $('terminal-original-log').download = 'original-terminal.bin';
}

function sizeTerminal() {
  fit.fit();
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'resize', cols: Math.min(500, Math.max(20, terminal.cols)), rows: Math.min(200, Math.max(5, terminal.rows)) }));
  }
}
new ResizeObserver(sizeTerminal).observe($('terminal'));

async function connectTerminal() {
  await terminalWrite;
  if (socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) return;
  if (currentRun?.library?.terminal_available === false || currentRun?.terminal_available === false) return;
  const url = new URL(apiURL('/v1/terminal'), location.href);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('after', String(terminalCursor));
  if (terminalSession) url.searchParams.set('session_id', terminalSession);
  const connection = new WebSocket(url);
  socket = connection;
  connection.onopen = () => { terminalStreamError = undefined; $('terminal-status').textContent = historyMode() ? '正在读取历史终端 · 只读' : '已连接 · 等待原生 TUI'; sizeTerminal(); };
  connection.onmessage = ({ data }) => {
    if (connection !== socket) return;
    let message;
    try { message = JSON.parse(data); } catch { $('terminal-status').textContent = '终端协议错误'; return; }
    if (message.type === 'restoring') { $('terminal-status').textContent = message.message; return; }
    if (message.type === 'ready' || message.type === 'status') {
      const status = message.status || message;
      if (status.session_id && terminalSession && status.session_id !== terminalSession) { terminalBootstrapKnown = undefined; terminalBootstrapText = ''; codexReplay.clearTerminalFailure(); }
      terminalSession = status.session_id || terminalSession;
      applyTerminalDiagnosis(status);
      terminalReplayTarget = status.cursor ?? terminalCursor;
      if (terminalCursor === 0 && status.replay?.kind === 'recovered_native_tui' && Number.isInteger(status.cols) && Number.isInteger(status.rows)) terminal.resize(status.cols, status.rows);
      terminalEnded = status.exit_code !== undefined && status.exit_code !== null;
      $('terminal-status').textContent = terminalReplayInfo?.kind === 'recovered_native_tui' ? terminalLabel()
        : status.exit_code !== undefined && status.exit_code !== null ? `原生 TUI 已退出 (${status.exit_code})` : terminalLabel();
      phaseReview.terminalUpdated();
    } else if (message.type === 'output') {
      terminalWrite = terminalWrite.then(() => new Promise((resolve) => {
        if (message.cursor <= terminalCursor) { resolve(); return; }
        if (message.after !== undefined && message.after > terminalCursor) {
          $('terminal-status').textContent = '字节流缺口 · 重新附着';
          terminalEnded = false;
          connection.close();
          resolve();
          return;
        }
        const bytes = Uint8Array.from(atob(message.data), (character) => character.charCodeAt(0)).slice(Math.max(0, terminalCursor - (message.after ?? terminalCursor)));
        terminalBootstrapText = (terminalBootstrapText + new TextDecoder().decode(bytes)).slice(-12000);
        if (terminalBootstrapKnown === undefined && terminalBootstrapText.includes('Error: account/read failed during TUI bootstrap:')) codexReplay.terminalFailure('原生终端在账户初始化时失败；可在“阶段记录”查看本次运行已保存的 Codex 消息、工具输入与结果。');
        terminal.write(bytes, () => {
          terminalCursor = message.cursor;
          if ((!terminalEnded || terminalReplayInfo?.kind === 'recovered_native_tui') && connection === socket && connection.readyState === WebSocket.OPEN) {
            $('terminal-status').textContent = terminalLabel();
          }
          if (terminalReplayInfo?.kind === 'recovered_native_tui' && terminalCursor >= terminalReplayTarget) requestAnimationFrame(sizeTerminal);
          resolve();
          phaseReview.terminalUpdated();
        });
      }));
    } else if (message.type === 'gap') {
      terminalWrite = terminalWrite.then(() => {
        markGap(`终端字节流缺口 · ${message.reason || '游标不可用'} · 从持久日志重新附着`);
        terminal.reset();
        terminalCursor = message.cursor ?? 0;
        terminalSession = undefined; terminalEnded = false; terminalBootstrapKnown = undefined; terminalBootstrapText = ''; codexReplay.clearTerminalFailure();
        connection.close();
      });
    } else if (message.type === 'error') {
      if (message.terminal) { terminalEnded = true; terminalStreamError = message.message; }
      $('terminal-status').textContent = message.message || message.error || '终端尚未就绪';
    } else if (message.type === 'exit') {
      terminalEnded = true;
      $('terminal-status').textContent = `原生 TUI 已退出 (${message.exit_code ?? '未知'})`;
    }
  };
  connection.onerror = () => { $('terminal-status').textContent = '终端连接异常'; };
  connection.onclose = () => {
    if (connection !== socket) return;
    if (terminalStreamError || (terminalEnded && terminalCursor >= terminalReplayTarget)) return;
    $('terminal-status').textContent = historyMode() ? '历史终端连接断开 · 正在重读' : '终端连接断开 · 正在附着';
    refreshRun();
    setTimeout(connectTerminal, 1500);
  };
}
$('terminal-bottom').onclick = () => terminal.scrollToBottom();
$('terminal-retry').onclick = async () => {
  await terminalWrite;
  const previous = socket; socket = undefined; previous?.close();
  terminal.reset(); terminalCursor = 0; terminalSession = undefined; terminalEnded = false;
  terminalStreamError = undefined; terminalBootstrapKnown = undefined; terminalBootstrapText = '';
  await connectTerminal();
};
function terminalLabel() {
  if (terminalStreamError) return terminalStreamError;
  if (terminalReplayInfo?.kind === 'native_resume_pending') return '正在恢复原生 TUI · 只读';
  if (terminalReplayInfo?.kind === 'recovered_native_tui') return terminalCursor < terminalReplayTarget ? '正在读取原生 TUI 归档恢复结果' : '原生 TUI · 会话归档恢复 · 只读';
  if (historyMode()) return terminalCursor < terminalReplayTarget ? '同步历史终端 · 只读回放' : '历史终端 · 只读回放';
  return terminalCursor < terminalReplayTarget ? '同步原生 TUI · 只读' : currentRun ? '实时 · 只读' : '终端已连接 · 核对运行模式';
}

function artifactURL(reference) {
  if (typeof reference !== 'string' || !reference.startsWith('artifact:')) return null;
  return apiURL(`/v1/artifacts/${encodeURIComponent(reference.slice('artifact:'.length))}`);
}

function addVideoExport(event) {
  for (const entry of event.payload?.videos || []) {
    const key = `${event.event_id}:${entry.camera_id}`;
    if (videoRecords.has(key)) continue;
    const record = { entry, event, identity: { ...event, ...event.payload } };
    const article = document.createElement('article'); article.className = 'video-record';
    article.dataset.exportEventId = event.event_id; article.dataset.cameraId = entry.camera_id;
    const title = document.createElement('h3');
    title.textContent = `${entry.camera_id}${entry.codec ? ` · ${entry.codec}` : ' · 原始导出'} · generation ${record.identity.env_generation} · ${record.identity.episode_id}`;
    const sampling = document.createElement('p'); sampling.className = 'meta';
    sampling.textContent = `${entry.frames ?? '未知'} 帧 · 编码 ${entry.fps ?? '未知'} fps · ${entry.timing || '采样策略未提供'} · 定位只读取记录的 PTS${entry.derivation?.lossy_transcode ? ' · 有损浏览器转码，原始文件另行保留' : ''}`;
    const video = document.createElement('video'); video.controls = true; video.playsInline = true;
    video.preload = 'none'; video.muted = true; video.setAttribute('aria-label', `${entry.camera_id} 已导出录像`);
    const source = artifactURL(entry.video_ref); if (source) video.src = source;
    const status = document.createElement('p'); status.className = 'meta video-media-status'; status.setAttribute('role', 'status');
    status.textContent = source ? '展开录像区后读取已导出的 MP4' : '录像 artifact 引用缺失';
    video.onloadeddata = () => { status.textContent = `MP4 已加载 · ${video.videoWidth}×${video.videoHeight} · 时长 ${video.duration.toFixed(3)} s`; };
    video.onseeked = () => { status.textContent = `已定位 · PTS ${video.currentTime.toFixed(3)} s`; };
    video.onerror = () => {
      status.textContent = `录像媒体读取失败 · code ${video.error?.code ?? '未知'}${video.error?.code === 4 ? ' · 浏览器不支持此格式或媒体不可用；可下载原文件' : ''}`;
      record.buttons?.forEach((button) => { button.disabled = true; });
    };
    const actionStatus = document.createElement('p'); actionStatus.className = 'meta video-action-status';
    const controls = document.createElement('div'); controls.className = 'video-actions';
    const buttons = ['动作开始帧', '动作结束帧'].map((label) => {
      const button = document.createElement('button'); button.textContent = label; button.disabled = true;
      controls.append(button); return button;
    });
    const download = document.createElement('a'); download.textContent = '下载此 MP4'; download.href = source || '#'; download.download = `${entry.camera_id}.mp4`;
    article.append(title, sampling, video, status, download, actionStatus, controls, jsonDetails('录制身份、采样与 frame mapping', {
      run_id: record.identity.run_id, episode_id: record.identity.episode_id, env_instance_id: record.identity.env_instance_id,
      env_generation: record.identity.env_generation, clock_domain: event.clock_domain, export_event_id: event.event_id,
      camera_id: entry.camera_id, video_ref: entry.video_ref, derived_from_ref: entry.derived_from_ref, derivation: entry.derivation, codec: entry.codec,
      frames: entry.frames, fps: entry.fps,
      timing: entry.timing, step_domain: entry.step_domain ?? event.step_domain ?? '未提供；定位需精确观测帧锚点', frame_mapping: entry.frame_mapping,
    }));
    Object.assign(record, { article, video, actionStatus, buttons });
    videoRecords.set(key, record);
    if (videoRecords.size > 96) {
      const oldest = videoRecords.keys().next().value, removed = videoRecords.get(oldest);
      removed.video.pause(); removed.article.remove(); videoRecords.delete(oldest);
    }
    const older = Array.from($('video-list').children).find((child) => child.dataset.exportEventId !== event.event_id);
    $('video-list').insertBefore(article, older || null);
    if (document.querySelector('.video-panel').open) video.load();
  }
  $('video-status').textContent = `${videoRecords.size} 路已导出录像 · 按 episode / generation 区分 · 原生播放器只读取记录`;
  updateVideoAction();
  phaseReview.setVideos(videoRecords.values());
}

function updateVideoAction() {
  for (const record of videoRecords.values()) {
    record.buttons.forEach((button) => { button.disabled = true; });
    delete record.article.dataset.mappedActionId;
    const selection = selectedVideoAction;
    if (!selection) { record.actionStatus.textContent = '选择动作证据后，按实际 frame mapping 定位'; continue; }
    const { action, result } = selection, request = action.request;
    const mapped = mapActionVideo(record, action, result);
    if (mapped.reason) { record.actionStatus.textContent = mapped.reason; continue; }
    const { start, end, domain, basis } = mapped;
    record.article.dataset.mappedActionId = request.action_id;
    record.article.dataset.mappedStartPts = start.pts_s; record.article.dataset.mappedEndPts = end.pts_s;
    record.actionStatus.textContent = `${request.action_id} · ${domain} ${start.sim_step}–${end.sim_step} → 记录 PTS ${start.pts_s}–${end.pts_s} s · ${basis === 'declared_step_domain' ? '已声明步域' : '终态帧锚点核对步域'}`;
    for (const [index, frame] of [start, end].entries()) {
      record.buttons[index].disabled = Boolean(record.video.error);
      record.buttons[index].onclick = () => {
        $('video-list').parentElement.open = true;
        const seek = () => { record.video.pause(); record.video.currentTime = frame.pts_s; };
        if (record.video.readyState >= 1) seek();
        else { record.video.addEventListener('loadedmetadata', seek, { once: true }); record.video.load(); }
      };
    }
  }
}
document.querySelector('.video-panel').ontoggle = (event) => {
  if (event.target.open) for (const record of videoRecords.values()) { record.video.preload = 'metadata'; if (record.video.readyState === 0) record.video.load(); }
};

function showObservation(observation) {
  displayedObservation = observation;
  const cameras = observation?.cameras || [];
  const previous = $('camera-select').value;
  $('camera-select').replaceChildren(...cameras.map((camera) => {
    const option = document.createElement('option');
    option.value = camera.camera_id;
    option.textContent = camera.camera_id + (camera.image_reconstruction ? ' · 视频重建（有损）' : '');
    return option;
  }));
  if (cameras.some((camera) => camera.camera_id === previous)) $('camera-select').value = previous;
  $('camera-heading').textContent = replay || historyMode() ? '机械臂 · 历史观察' : '机械臂 · 实时观察';
  $('return-live').textContent = historyMode() ? '返回最新记录' : '返回实时';
  $('return-live').hidden = !replay;
  showCamera();
  syncTimeline();
}

function showCamera() {
  const camera = displayedObservation?.cameras?.find((item) => item.camera_id === $('camera-select').value);
  const url = artifactURL(camera?.rgb_ref);
  const image = $('camera-image');
  image.hidden = true;
  $('camera-empty').hidden = false;
  if (url) {
    $('camera-empty').textContent = '正在读取所选帧媒体';
    image.onload = () => {
      image.hidden = image.naturalWidth === 0;
      $('camera-empty').hidden = image.naturalWidth > 0;
      if (!image.naturalWidth) $('camera-empty').textContent = '媒体解码失败 · 未沿用上一观察画面';
    };
    image.src = url + '?view=image';
    image.alt = `${camera.camera_id} · ${displayedObservation.observation_id} · ${camera.frame_id}`;
    if (image.complete && image.naturalWidth > 0) image.onload();
  } else {
    $('camera-image').removeAttribute('src');
    $('camera-empty').textContent = displayedObservation ? '该观察没有可访问的 RGB 媒体' : '等待 RoboDojo 发布观察';
  }
  const observation = displayedObservation;
  $('frame-meta').textContent = observation
    ? `${observation.observation_id} · generation ${observation.env_generation} · ${observation.step_domain || '步域未提供'} ${observation.sim_step ?? '未知'} · ${camera?.frame_id || '画面缺失'}${camera?.image_reconstruction ? ' · 视频重建（有损）' : ''}`
    : '观察与媒体将按 ID 关联';
}
$('camera-image').onerror = () => {
  $('camera-image').hidden = true;
  $('camera-empty').hidden = false;
  $('camera-empty').textContent = '媒体读取失败 · 未沿用上一观察画面';
};
$('camera-select').onchange = showCamera;
$('return-live').onclick = () => { replay = false; showObservation(latestObservation); };
$('close-inspector').onclick = () => { $('inspector').hidden = true; };

let observationRefreshing = false;
async function refreshObservation() {
  if (observationRefreshing) return;
  observationRefreshing = true;
  try {
    const observation = await getJSON('/v1/observations/latest');
    if (!observation?.observation_id) return;
    if (latestObservation?.observation_id === observation.observation_id) return;
    latestObservation = observation;
    if (!replay) showObservation(observation);
  } catch (error) {
    if (!displayedObservation) $('camera-empty').textContent = `观察暂不可用 · ${error.message}`;
  } finally { observationRefreshing = false; }
}

const eventLabels = {
  'observation.available': '观察已发布',
  'observation.delivered': '观察交付记录',
  'action.requested': 'EE 动作请求', 'action.accepted': '动作已接受',
  'action.started': '动作执行中', 'action.progress': '动作执行中',
  'action.completed': '动作已完成', 'action.failed': '动作失败',
  'action.rejected': '动作被拒绝', 'action.stop_requested': '请求停止动作',
  'action.stopped': '动作停止已确认', 'action.status_unknown': '动作状态待核实',
  'scene.updated': '内部场景已更新', 'scene.invalidated': '内部场景失效',
  'job.started': '内部预演开始', 'job.completed': '内部预演已完成',
  'job.failed': '内部预演失败', 'job.cancelled': '内部预演已取消',
  'run.stopped': '运行停止已确认', 'run.stop_requested': '正在请求停止运行',
  'codex.error': '{agent} 原生错误', 'codex.turn/started': '{agent} 轮次开始',
  'codex.turn/completed': '{agent} 轮次结束', 'codex.connection_lost': '{agent} 上游断开',
  'codex.history_recovered': '{agent} 历史已对账',
  'observer.video_exported': '观察者录像已导出',
};

// Backend-neutral lookup: agent.* aliases resolve to the codex.* entries; `{agent}` is the run's backend name.
const labelFor = (type) => (eventLabels[canonicalEventType(type)] || '').replaceAll('{agent}', agentName()) || undefined;
const hasLabel = (type) => Boolean(eventLabels[canonicalEventType(type)]);
function eventLabel(event) {
  if (event.type !== 'observation.delivered') return labelFor(event.type) || event.type;
  if (event.payload?.delivered_to_model === true) return 'Astra 模型已接收观察';
  if (event.payload?.audience === 'actor_program' || event.correlation?.program_id) return '程序已接收观察';
  return '观察交付记录 · 受众待核实';
}

function receiveEvent(event) {
  if (seenEvents.has(event.event_id) || (event.server_seq && event.server_seq <= eventCursor)) return;
  if (eventCursor && event.server_seq > eventCursor + 1) markGap(`事件接收序号 ${eventCursor + 1}–${event.server_seq - 1} 缺失 · 正在核对当前记录`);
  seenEvents.add(event.event_id);
  if (seenEvents.size > 5000) seenEvents.delete(seenEvents.values().next().value);
  eventCursor = Math.max(eventCursor, event.server_seq || 0);
  if (!historicalWindow && phaseReview.receive(event)) {
    eventsWindowed = true;
    updateRecordsWindow();
  }
  $('events-status').textContent = `${historyMode() ? '历史记录' : currentRun ? '实时' : '正在核对记录'} · 接收序号 ${eventCursor}`;
  if (event.type === 'observation.available' && !historyMode()) refreshObservation();
  if (event.type === 'observer.video_exported' && !historicalWindow) addVideoExport(event);
  if (event.type === 'observer.evaluated' || event.type?.startsWith('run.')) refreshTaskReview();
  if (canonicalEventType(event.type) === 'codex.connection_lost') markGap(`${agentName()} 上游连接丢失 · 原生 delta 历史不完整，需独立对账`);
  if (canonicalEventType(event.type) === 'codex.history_recovered' && event.payload?.delta_history_complete === false) markGap(`${agentName()} 历史已对账 · 终态记录可恢复，缺失的 delta 历史仍不完整`);
  scheduleRecords();
  if (!hasLabel(event.type) && !event.type?.startsWith('action.') && !event.type?.startsWith('job.')) return;
  const correlation = event.correlation || {};
  const key = correlation.action_id ? `action:${correlation.action_id}` : (correlation.job_id ? `job:${correlation.job_id}` : event.event_id);
  const existing = rows.get(key);
  const button = existing?.button || document.createElement('button');
  button.dataset.actionId = correlation.action_id || '';
  button.dataset.eventType = event.type;
  const sequence = document.createElement('span');
  sequence.className = 'event-seq';
  sequence.textContent = `#${event.server_seq}`;
  const label = document.createElement('span');
  label.className = 'event-label';
  label.textContent = `${eventLabel(event)} · ${correlation.action_id || correlation.observation_id || correlation.job_id || event.event_id}`;
  const state = document.createElement('span');
  state.className = `event-state ${event.type.split('.').at(-1)}`;
  state.textContent = isNativeAgentEvent(event.type) ? `原生 ${agentName()}` : event.world_kind === 'internal' ? '内部结果' : '被测世界';
  button.replaceChildren(sequence, label, state);
  button.onclick = () => inspectEvent(event, button);
  rows.set(key, { button, event });
  if (!existing) {
    $('activity').querySelector('.empty')?.remove();
    const item = document.createElement('li');
    item.append(button);
    $('activity').prepend(item);
  }
  // ponytail: retain 250 visible rows; use server pagination if a full event browser is needed.
  if (rows.size > 250) {
    const oldest = rows.keys().next().value;
    rows.get(oldest).button.parentElement.remove();
    rows.delete(oldest);
  }
}

async function inspectEvent(event, button) {
  const version = ++selectionVersion;
  selectedVideoAction = null; updateVideoAction();
  selectedEvent?.classList.remove('selected');
  selectedEvent = button;
  button.classList.add('selected');
  $('inspector').hidden = false;
  $('inspector-title').textContent = eventLabel(event);
  const correlation = event.correlation || {};
  $('inspector-summary').textContent = `接收序号 ${event.server_seq} · ${event.world_kind === 'internal' ? '内部结果' : '被测世界'} · ${agentName()} item ${correlation.item_id || '未关联'} · 工具调用 ${correlation.tool_call_id || '未关联'}${event.type === 'observation.delivered' ? ' · 交付受众、通道、原始图像哈希与变换见交付记录' : ''}`;
  $('observation-links').replaceChildren();
  $('trace-records').replaceChildren();
  $('associated-native').replaceChildren();
  const records = { event };
  if (event.payload?.raw_ref) {
    try { records.native_event = await getJSON(artifactURL(event.payload.raw_ref)); }
    catch (error) { records.native_event_unavailable = error.message; }
    if (version !== selectionVersion) return;
  }
  try {
    if (correlation.action_id) records.action = await getJSON(`/v1/actions/${encodeURIComponent(correlation.action_id)}`);
    if (version !== selectionVersion) return;
    const action = records.action;
    const inputId = action?.request?.based_on_observation_id || action?.based_on_observation_id || correlation.observation_id;
    const resultId = action?.receipt?.result_observation_id || action?.result_observation_id || event.payload?.result_observation_id;
    const beforeId = action?.receipt?.execution_start_observation_id;
    const inputLabel = correlation.action_id ? '动作输入观察' : '关联观察';
    for (const [title, id] of [[inputLabel, inputId], ['执行前实测观察', beforeId], ['执行后观察', resultId]]) {
      if (!id) continue;
      const link = document.createElement('button');
      link.textContent = `${title} · ${id}`;
      link.onclick = () => selectObservation(id);
      $('observation-links').append(link);
    }
    if (action) renderTraces(action, version);
    if (correlation.item_id) renderAssociatedNative(correlation.item_id, version);
    if (inputId) {
      records.input_observation = await getJSON(`/v1/observations/${encodeURIComponent(inputId)}`);
      if (version !== selectionVersion) return;
      replay = true;
      showObservation(records.input_observation);
    } else if (correlation.action_id) {
      replay = true;
      showObservation(null);
      $('camera-empty').textContent = '该动作未关联输入观察';
    }
    if (action) {
      if (resultId) {
        try { records.result_observation = await getJSON(`/v1/observations/${encodeURIComponent(resultId)}`); }
        catch (error) { records.video_mapping_unavailable = error.message; }
        if (version !== selectionVersion) return;
      }
      selectedVideoAction = { action, result: records.result_observation }; updateVideoAction();
    }
  } catch (error) {
    records.unavailable = error.message;
    if (version !== selectionVersion) return;
    replay = true;
    showObservation(null);
    $('camera-empty').textContent = '关联观察或动作记录不可用';
  }
  $('inspector-data').textContent = JSON.stringify(records, null, 2);
}

async function selectObservation(id) {
  const version = ++selectionVersion;
  try {
    const observation = await getJSON(`/v1/observations/${encodeURIComponent(id)}`);
    if (version !== selectionVersion) return;
    replay = true;
    showObservation(observation);
    syncTimeline();
  } catch (error) {
    if (version !== selectionVersion) return;
    replay = true;
    showObservation(null);
    $('camera-empty').textContent = `历史观察不可用 · ${error.message}`;
  }
}

const events = new EventSource(apiURL(`/v1/events?after=${eventCursor}`));
events.onopen = () => { $('events-status').textContent = `${historyMode() ? '历史记录' : currentRun ? '实时' : '正在核对记录'} · 接收序号 ${eventCursor}`; };
events.onerror = () => { $('events-status').textContent = '事件连接断开 · 正在恢复'; };
const parseEvent = ({ data }) => {
  try { receiveEvent(JSON.parse(data)); } catch (error) { $('events-status').textContent = `事件读取失败 · ${error.message}`; }
};
events.onmessage = parseEvent;
events.addEventListener('event', parseEvent);
events.addEventListener('gap', ({ data }) => {
  let gap = {}; try { gap = JSON.parse(data); } catch { /* Display the explicit missing state. */ }
  markGap(`事件历史存在缺口 · ${gap.reason || '保留范围不可用'} · 当前接收序号 ${gap.current_seq ?? '未知'}`);
  eventCursor = gap.current_seq ?? eventCursor;
  refreshRun(); refreshRecords();
});

async function refreshRun() {
  try {
    const run = await getJSON('/v1/run');
    const initialLoad = !currentRun;
    const previousMode = historyMode();
    const previousStatus = currentRun?.status;
    currentRun = run;
    if (initialLoad || previousMode !== historyMode() || previousStatus !== run.status) refreshTaskReview();
    document.body.classList.toggle('history-review', Boolean(historyMode()));
    if (initialLoad) {
      $('still-observations').open = !historyMode() && videoRecords.size === 0;
      codexReplay.setTab('terminal');
    }
    $('run-status').textContent = `${run.status || run.state || '已连接'} · ${run.run_id || run.run?.run_id || '运行'}`;
    $('run-mode').textContent = historyMode() ? '只读回放' : readOnly() ? '实时记录 · 只读' : '实时监看';
    $('events-status').textContent = `${historyMode() ? '历史记录' : '实时'} · 接收序号 ${eventCursor}`;
    $('camera-heading').textContent = replay || historyMode() ? '机械臂 · 历史观察' : '机械臂 · 实时观察';
    $('return-live').textContent = historyMode() ? '返回最新记录' : '返回实时';
    updateTerminalNote();
    document.querySelector('.terminal-panel').setAttribute('aria-label', historyMode() ? `${agentName()} 阶段回放与原始终端` : `${agentName()} 阶段记录与实时终端`);
    applyAgentNames(document, agentName());
    const noTerminal = terminalAbsent(run);
    $('codex-tab-terminal').hidden = noTerminal; // backend without a native TUI (`terminal: null`): records tab only
    if (run.library?.terminal_available === false || run.terminal_available === false || noTerminal) {
      $('codex-tab-terminal').disabled = true;
      codexReplay.setTab('records');
      socket?.close();
    } else {
      $('codex-tab-terminal').disabled = false;
      if (!terminalEnded && socket?.readyState === WebSocket.OPEN) $('terminal-status').textContent = terminalLabel();
      else if (!terminalEnded && socket?.readyState !== WebSocket.CONNECTING) connectTerminal();
    }
    const noCodex = (run.library?.terminal_available === false || run.terminal_available === false) && !run.thread_id;
    document.querySelector('.task-panel').hidden = noCodex;
    document.querySelector('.native-panel').hidden = noCodex;
    applyTerminalDiagnosis(run.terminal);
    if (terminalReplayInfo || terminalStreamError) $('terminal-status').textContent = terminalLabel();
    else if (run.terminal?.exit_code !== undefined && run.terminal.exit_code !== null) {
      $('terminal-status').textContent = `原生 TUI 已退出 (${run.terminal.exit_code})`;
    }
    $('run-data').textContent = JSON.stringify(run, null, 2);
    $('turn-state').textContent = `${agentName()} 轮次 · ${agentStatus(run) || '未知'} · ${run.turn_id || '尚无轮次'}`;
    $('motion-state').textContent = `机器人动作 · ${run.motion?.status || run.motion_status || '未知'} · ${run.motion?.active_action || run.active_action || '无活动动作'}`;
    updateOwnership();
  } catch (error) { $('run-status').textContent = `运行连接异常 · ${error.message}`; $('run-mode').textContent = '连接中断'; }
}

let taskReviewRefreshing = false, taskReviewAgain = false;
async function refreshTaskReview() {
  if (taskReviewRefreshing) { taskReviewAgain = true; return; } taskReviewRefreshing = true;
  try { renderTaskReview($('run-task-review'), await getJSON('/v1/task-review')); }
  catch (error) { renderTaskReview($('run-task-review'), null, `评分证据暂不可用 · ${error.message}`); }
  finally { taskReviewRefreshing = false; if (taskReviewAgain) { taskReviewAgain = false; refreshTaskReview(); } }
}
setInterval(() => { if (!historyMode()) refreshTaskReview(); }, 30000);

function markGap(message) {
  gapMessages.add(message);
  if (gapMessages.size > 5) gapMessages.delete(gapMessages.values().next().value);
  $('connection-gap').hidden = false;
  $('connection-gap').textContent = [...gapMessages].join('\n');
}

function post(path, value) {
  return getJSON(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
}

function control(command, fields = {}) {
  return post('/v1/control', { request_id: crypto.randomUUID(), command, client_id: clientId, writer_token: writerToken, ...fields });
}

function updateOwnership() {
  const controller = currentRun?.controller;
  const owned = !readOnly() && Boolean(writerToken && controller?.owner_client_id === clientId);
  if (readOnly()) writerToken = undefined;
  if (writerToken && controller && controller.owner_client_id !== clientId) writerToken = undefined;
  $('input-status').textContent = readOnly() ? currentRun ? historyMode() ? '只读回放 · 输入控制已关闭' : '实时记录 · 输入控制已关闭' : '正在核对运行模式' : owned ? '本窗口持有输入控制' : controller?.owner_client_id ? '另一窗口持有输入控制 · 只读' : '监看模式 · 可取得输入控制';
  $('claim-input').hidden = owned || Boolean(historyMode());
  $('claim-input').disabled = readOnly() || Boolean(controller?.owner_client_id && controller.owner_client_id !== clientId);
  $('release-input').hidden = !owned;
  $('task-prompt').disabled = readOnly();
  $('stop').disabled = readOnly();
  if (readOnly()) $('task-status').textContent = currentRun ? '已保存的任务与交互保留在原生记录中' : '正在核对运行模式';
  document.querySelectorAll('.scene-panel form button,.scene-panel form textarea').forEach((input) => { input.disabled = readOnly(); });
  $('scene-snapshot').disabled = readOnly() || !$('scene-anchor').value;
  $('submit-task').disabled = !owned || !currentRun?.accepting_actions || agentStatus(currentRun) === 'running' || Boolean(currentRun?.motion?.active_action || currentRun?.active_action);
  document.querySelectorAll('#pending-interactions form button,#pending-interactions form input,#pending-interactions form select,#pending-interactions form textarea').forEach((input) => { input.disabled = !owned; });
}

$('claim-input').onclick = async () => {
  try {
    const result = await control('claim_input');
    writerToken = result.writer_token;
    await refreshRun();
  } catch (error) { $('input-status').textContent = `取得输入控制失败 · ${error.message}`; }
};
$('release-input').onclick = async () => {
  try { await control('release_input'); writerToken = undefined; await refreshRun(); }
  catch (error) { $('input-status').textContent = `释放失败 · ${error.message}`; }
};
$('task-form').onsubmit = async (event) => {
  event.preventDefault();
  const prompt = $('task-prompt').value.trim();
  if (!prompt || !writerToken) return;
  if (taskAttempt?.prompt !== prompt) taskAttempt = { request_id: crypto.randomUUID(), prompt };
  $('submit-task').disabled = true;
  $('task-status').textContent = '正在提交当前会话任务';
  try {
    const result = await control('prompt', taskAttempt);
    $('task-status').textContent = `已提交 · thread ${result.thread_id} · turn ${result.turn_id}`;
    $('task-prompt').value = '';
    taskAttempt = undefined;
    await refreshRun();
  } catch (error) {
    $('task-status').textContent = `任务提交未确认 · ${error.message} · 原内容重试沿用同一请求 ID`;
    updateOwnership();
  }
};
setInterval(async () => {
  if (!writerToken) return;
  try { await control('renew_input'); }
  catch (error) { writerToken = undefined; $('input-status').textContent = `输入租约未确认 · ${error.message}`; updateOwnership(); }
}, 10000);
window.addEventListener('pagehide', () => {
  if (!writerToken) return;
  fetch(apiURL('/v1/control'), { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command: 'release_input', request_id: crypto.randomUUID(), client_id: clientId, writer_token: writerToken }) }).catch(() => {});
});

function jsonDetails(title, value, open = false) {
  const details = document.createElement('details');
  details.open = open;
  const summary = document.createElement('summary');
  summary.textContent = title;
  const pre = document.createElement('pre');
  pre.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  details.append(summary, pre);
  return details;
}

async function renderTraces(action, version) {
  const container = $('trace-records');
  container.append(jsonDetails('请求 · EE 原始目标与路径', action.request));
  for (const [label, key] of [['规划', 'planned_trace_ref'], ['下发', 'dispatched_trace_ref'], ['实际实测', 'actual_trace_ref'], ['执行前状态', 'before_state_ref']]) {
    const reference = action.receipt?.[key];
    const details = jsonDetails(`${label} · ${reference || '未记录'}`, reference ? '展开后读取原始记录' : '未记录；不使用请求轨迹代替');
    container.append(details);
    let loaded = false;
    details.ontoggle = async () => {
      if (!details.open || loaded || !reference) return;
      loaded = true;
      try {
        const data = await getJSON(artifactURL(reference));
        if (!details.isConnected) return;
        details.querySelector('pre').textContent = JSON.stringify(data, null, 2);
      } catch (error) { details.querySelector('pre').textContent = `记录不可用 · ${error.message}`; }
    };
  }
}

async function renderAssociatedNative(id, version) {
  const marker = document.createElement('p');
  marker.textContent = `正在读取原生 item ${id}`;
  $('associated-native').append(marker);
  try {
    const item = await getJSON(`/v1/items/${encodeURIComponent(id)}`);
    if (!marker.isConnected) return;
    marker.replaceWith(nativeDetails(item, true));
  } catch (error) {
    if (marker.isConnected) marker.textContent = `原生 item ${id} 未记录或不可用 · ${error.message}`;
  }
}

function nativeDetails(item, open = false) {
  const details = jsonDetails(`${item.type || '原生 item'} · ${item.id} · ${item.status || '状态未提供'}`, item, open);
  const rawRecord = details.querySelector('pre');
  const fields = document.createElement('p'); fields.className = 'record-fields';
  for (const [label, key] of [['thread', 'thread_id'], ['turn', 'turn_id'], ['工具', 'tool'], ['命令', 'command']]) {
    if (item[key] === undefined) continue;
    const span = document.createElement('span'); span.textContent = `${label} · ${typeof item[key] === 'string' ? item[key] : JSON.stringify(item[key])}`; fields.append(span);
  }
  details.insertBefore(fields, rawRecord);
  if (item.text) {
    const text = document.createElement('pre');
    text.textContent = item.text;
    details.insertBefore(text, rawRecord);
  }
  for (const [label, key] of [['工具参数','arguments'],['工具结果','result'],['可见命令输出','aggregatedOutput'],['错误','error'],['文件与代码差异','changes']]) {
    if (item[key] !== undefined) details.insertBefore(jsonDetails(label, item[key]), rawRecord);
  }
  for (const ref of item.raw_refs || []) {
    const url = artifactURL(typeof ref === 'string' ? ref : ref.raw_ref);
    if (!url) continue;
    const link = document.createElement('a');
    link.href = url; link.target = '_blank'; link.rel = 'noopener'; link.textContent = '原始原生事件';
    details.append(link, document.createTextNode(' '));
  }
  return details;
}

function renderNativeItems(items, cursor) {
  const expanded = new Set([...$('native-items').querySelectorAll('details[open]')].map((node) => node.parentElement.dataset.itemId));
  $('native-items').replaceChildren(...items.map((item) => {
    const li = document.createElement('li');
    li.dataset.itemId = item.id;
    li.append(nativeDetails(item, expanded.has(item.id)));
    return li;
  }));
  $('native-status').textContent = `${items.length} 个原生 item · 接收序号 ${cursor ?? '未提供'} · 状态与机器人运动分别判断`;
}

function renderRequests(requests) {
  const existing = new Map([...$('pending-interactions').children].map((node) => [node.dataset.requestId, node]));
  $('pending-interactions').replaceChildren(...requests.filter((request) => ['pending', 'responding', 'unknown'].includes(request.status)).map((request) => {
    const signature = JSON.stringify(request);
    const previous = existing.get(String(request.request_id));
    if (previous?.dataset.signature === signature) return previous;
    const section = document.createElement('section');
    section.className = 'pending-request';
    section.dataset.requestId = String(request.request_id);
    section.dataset.signature = signature;
    const title = document.createElement('h3');
    title.textContent = `${request.method?.includes('requestApproval') ? '原生审批' : '原生输入请求'} · ${request.request_id}`;
    const meta = document.createElement('p');
    meta.className = 'meta';
    meta.textContent = `${request.method} · native RPC ID ${request.rpc_id ?? request.request_id} · turn ${request.turn_id || '未提供'} · item ${request.item_id || '未提供'} · 到期 ${request.deadline_at || '未提供'}`;
    if (request.status !== 'pending') {
      const state = document.createElement('p'); state.className = 'meta';
      state.textContent = request.status === 'responding' ? '回复已交给网关 · 原生送达仍在等待确认' : `原生回复状态未知 · ${request.response_error || request.reason || '没有成功发送回执'}`;
      section.append(title, meta, state, jsonDetails('原生请求与参数', request));
      return section;
    }
    const form = document.createElement('form');
    const questions = request.params?.questions;
    const inputs = [];
    let responseInput;
    if (Array.isArray(questions)) {
      for (const question of questions) {
        const label = document.createElement('label');
        const id = `question-${crypto.randomUUID()}`;
        label.htmlFor = id; label.textContent = question.question || question.header || question.id;
        const input = document.createElement('input');
        input.id = id; input.type = question.isSecret ? 'password' : 'text'; input.required = true;
        const datalist = document.createElement('datalist');
        datalist.id = `${id}-choices`; input.setAttribute('list', datalist.id);
        for (const option of question.options || []) {
          const choice = document.createElement('option'); choice.value = option.label; choice.textContent = option.description || ''; datalist.append(choice);
        }
        inputs.push([question.id, input]); form.append(label, input, datalist);
      }
    } else if (request.method?.includes('requestApproval') && request.method !== 'item/permissions/requestApproval') {
      responseInput = document.createElement('select');
      responseInput.setAttribute('aria-label', '原生审批决定');
      for (const decision of request.params?.availableDecisions || ['decline', 'accept', 'cancel']) {
        const option = document.createElement('option'); option.value = JSON.stringify(decision); option.textContent = typeof decision === 'string' ? decision : JSON.stringify(decision); responseInput.append(option);
      }
      form.append(responseInput);
    } else {
      responseInput = document.createElement('textarea'); responseInput.value = request.method === 'item/permissions/requestApproval' ? '{"permissions":{},"scope":"turn"}' : '{}'; responseInput.setAttribute('aria-label', '原生响应 JSON'); form.append(responseInput);
    }
    const button = document.createElement('button'); button.type = 'submit'; button.textContent = '回复该原生请求'; form.append(button);
    const status = document.createElement('p'); status.className = 'meta'; status.setAttribute('role', 'status'); form.append(status);
    let attempt;
    form.onsubmit = async (event) => {
      event.preventDefault();
      try {
        const response = Array.isArray(questions) ? { answers: Object.fromEntries(inputs.map(([id, input]) => [id, { answers: [input.value] }])) }
          : responseInput.tagName === 'SELECT' ? { decision: JSON.parse(responseInput.value) } : JSON.parse(responseInput.value);
        const signature = JSON.stringify(response);
        if (attempt?.signature !== signature) attempt = { signature, request_id: crypto.randomUUID() };
        button.disabled = true;
        const result = await control('reply', { request_id: attempt.request_id, native_request_id: request.request_id, response });
        status.textContent = result.status === 'resolved' ? '已回复该原生请求' : `网关已接收回复 · 原生送达 ${result.status || '尚未确认'}`;
        await refreshRecords();
      } catch (error) { status.textContent = `回复未确认 · ${error.message}`; updateOwnership(); }
    };
    section.append(title, meta, jsonDetails('原生请求与参数', request), form);
    return section;
  }));
  updateOwnership();
}

function syncTimeline() {
  const index = observations.findIndex((item) => item.observation_id === displayedObservation?.observation_id);
  const range = $('frame-timeline'); range.max = Math.max(0, observations.length - 1); range.disabled = observations.length === 0;
  if (index >= 0) range.value = index;
  $('frame-previous').disabled = index <= 0;
  $('frame-next').disabled = index < 0 || index >= observations.length - 1;
  const observation = displayedObservation;
  $('timeline-meta').textContent = observation ? `${index >= 0 ? `${index + 1}/${observations.length} · ` : ''}${observation.observation_id} · generation ${observation.env_generation} · ${observation.step_domain || '步域未提供'} ${observation.sim_step ?? '未知'} · 动作 ${observation.last_action_id || '未关联'} · 标定 ${observation.calibration_id || '未提供；不作像素投影'}` : '没有选中可用观察';
}

async function inspectTimeline(index) {
  const observation = observations[index];
  if (!observation) return;
  const expectedVersion = selectionVersion + 1;
  await selectObservation(observation.observation_id);
  if (expectedVersion !== selectionVersion) return;
  const version = selectionVersion;
  $('inspector').hidden = false;
  $('inspector-title').textContent = '观测与媒体记录';
  $('inspector-summary').textContent = `精确观察 ${observation.observation_id} · ${observation.step_domain || '步域未提供'} ${observation.sim_step ?? '未知'} · 不按墙钟补造同步关系`;
  $('trace-records').replaceChildren(); $('associated-native').replaceChildren(); $('observation-links').replaceChildren();
  const records = { observation };
  if (observation.last_action_id) {
    try {
      records.action = await getJSON(`/v1/actions/${encodeURIComponent(observation.last_action_id)}`);
      if (version !== selectionVersion) return;
      renderTraces(records.action, version);
    } catch (error) { records.action_unavailable = error.message; }
  }
  if (version === selectionVersion) $('inspector-data').textContent = JSON.stringify(records, null, 2);
}
$('frame-timeline').oninput = () => inspectTimeline(Number($('frame-timeline').value));
$('frame-previous').onclick = () => inspectTimeline(Number($('frame-timeline').value) - 1);
$('frame-next').onclick = () => inspectTimeline(Number($('frame-timeline').value) + 1);

function scheduleRecords() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refreshRecords, 100);
}
let recordsRefreshing = false, recordsAgain = false, recordsLoaded = false, recordsRetryAttempts = 0, recordsRetryTimer;
function updateRecordsWindow() {
  $('records-older').hidden = !incrementalRecords || (!recordCaches.items.dropped && !(eventsWindowed && nativeItems.length));
  const items = incrementalRecords && recordCaches.items.dropped ? '显示最近 2000 条记录；较早记录可按需读取。' : '';
  const events = eventsWindowed ? '阶段事件仅显示最近 5000 条；其余事件请查看完整事件证据。' : '';
  $('records-window').textContent = items + events;
  $('records-events').hidden = !eventsWindowed;
}
$('records-older').onclick = async () => {
  try {
    const before = historicalWindow ? historyBefore : recordCaches.items.dropped ? recordCaches.items.firstPosition : undefined;
    const page = await getJSON(`/v1/records?mode=history${Number.isFinite(before) ? `&before=${before}` : ''}`);
    historicalWindow = true; historyBefore = page.before;
    nativeItems = page.items.map((entry) => entry.value);
    for (const record of videoRecords.values()) { record.video.pause(); record.article.remove(); }
    videoRecords.clear();
    for (const event of page.videos || []) addVideoExport(event);
    phaseReview.replace(page.events, nativeItems);
    codexReplay.setItems(nativeItems); renderNativeItems(nativeItems);
    $('records-older').hidden = !page.has_more; $('records-latest').hidden = false;
    $('records-window').textContent = `历史窗口 · ${nativeItems.length} 条记录 · 较早记录切换原生记录页${page.events_truncated ? ' · 关联事件仅显示前 5000 条；其余关联事件请查看完整事件证据' : ''}`;
    $('records-events').hidden = false;
  } catch (error) { $('records-window').textContent = `历史窗口读取失败 · ${error.message}`; }
};
$('records-latest').onclick = () => {
  // Reload restores the persistent SSE cursor stream, never starts or controls a run.
  location.reload();
};
$('phase-retry').addEventListener('click', () => { recordsRetryAttempts = 0; clearTimeout(recordsRetryTimer); refreshRecords(); refreshTaskReview(); });
async function refreshRecords() {
  if (historicalWindow) return;
  if (recordsRefreshing) { recordsAgain = true; return; }
  recordsRefreshing = true;
  if (incrementalRecords) {
    try {
      const query = new URLSearchParams(Object.entries(recordCaches).map(([name, cache]) => [name, cache.cursor]));
      const value = await getJSON(`/v1/records?${query}`, { signal: AbortSignal.timeout(15000) });
      if (historicalWindow) return;
      const nextObservations = recordCaches.observations.apply(value.observations);
      const nextItems = recordCaches.items.apply(value.items);
      const nextRequests = recordCaches.requests.apply(value.requests);
      if (!recordsLoaded || value.observations.entries.length || value.observations.reset) {
        observations = nextObservations;
        phaseReview.setObservations(observations); syncTimeline();
      }
      if (!recordsLoaded || value.items.entries.length || value.items.reset) {
        nativeItems = nextItems;
        renderNativeItems(nativeItems, value.event_cursor); codexReplay.setItems(nativeItems);
        phaseReview.setItems(nativeItems, value.event_cursor);
      }
      if (!recordsLoaded || value.requests.entries.length || value.requests.reset) {
        pendingRequests = nextRequests; renderRequests(pendingRequests);
      }
      recordsLoaded = true;
      $('native-status').textContent = `${nativeItems.length} 个原生 item · 接收序号 ${value.event_cursor}`;
      updateRecordsWindow();
      recordsRetryAttempts = 0;
      if (Object.keys(recordCaches).some((name) => value[name].has_more)) recordsAgain = true;
      return;
    } catch (error) {
      if (error.status === 404) incrementalRecords = false;
      else {
        $('native-status').textContent = `结构化记录不可用 · ${error.message}`;
        codexReplay.unavailable(error.message);
        if (recordsRetryAttempts < 3) recordsRetryTimer = setTimeout(refreshRecords, 1000 * 2 ** recordsRetryAttempts++);
        return;
      }
    } finally {
      recordsRefreshing = false;
      if (recordsAgain) { recordsAgain = false; scheduleRecords(); }
    }
    recordsRefreshing = true;
  }
  const results = await Promise.allSettled([getJSON('/v1/observations'), getJSON('/v1/items'), getJSON('/v1/requests')]);
  results.forEach((result, index) => {
    if (result.status === 'rejected') {
      if (index === 1) { $('native-status').textContent = `结构化记录不可用 · ${result.reason.message}`; codexReplay.unavailable(result.reason.message); }
      if (index === 0) $('timeline-meta').textContent = `观测索引不可用 · ${result.reason.message}`;
      return;
    }
    const value = result.value;
    if (index === 0) { observations = value.observations || []; phaseReview.setObservations(observations); syncTimeline(); }
    if (index === 1) {
      const currentItems = value.items || [];
      if (JSON.stringify(currentItems) !== JSON.stringify(nativeItems)) { nativeItems = currentItems; renderNativeItems(nativeItems, value.cursor); }
      $('native-status').textContent = `${nativeItems.length} 个原生 item · 接收序号 ${value.cursor ?? '未提供'} · 状态与机器人运动分别判断`;
      codexReplay.setItems(nativeItems);
      phaseReview.setItems(nativeItems, value.cursor);
    }
    if (index === 2 && JSON.stringify(value.requests) !== JSON.stringify(pendingRequests)) { pendingRequests = value.requests || []; renderRequests(pendingRequests); }
  });
  updateRecordsWindow();
  recordsRefreshing = false;
  clearTimeout(recordsRetryTimer);
  if (results.slice(0, 2).some((result) => result.status === 'rejected')) {
    if (recordsRetryAttempts < 3) recordsRetryTimer = setTimeout(refreshRecords, 1000 * 2 ** recordsRetryAttempts++);
  } else recordsRetryAttempts = 0;
  if (recordsAgain) { recordsAgain = false; scheduleRecords(); }
}

async function refreshScene() {
  try {
    const [head, versions] = await Promise.all([getJSON('/v1/scene/head'), getJSON('/v1/scene/versions')]);
    sceneVersions = versions.versions || [];
    const previous = $('scene-version').value;
    $('scene-version').replaceChildren(...sceneVersions.map((value) => {
      const option = document.createElement('option'); option.value = value.scene_version ?? value; option.textContent = `版本 ${option.value}`; return option;
    }));
    $('scene-version').value = sceneVersions.some((value) => String(value.scene_version ?? value) === previous) ? previous : String(head.scene_version ?? head.head ?? '');
    $('scene-snapshot').disabled = readOnly() || !$('scene-version').value;
    $('scene-status').textContent = `内部几何场景 · head ${head.scene_version ?? head.head ?? '尚无版本'} · RGB 重建 ${head.capabilities?.rgb_reconstruction || '按实际工具能力'} · 物理预演未接入`;
    $('scene-data').textContent = JSON.stringify(head, null, 2);
    if ($('scene-version').value) await loadSceneVersion();
  } catch (error) { $('scene-status').textContent = `局部场景暂不可用 · ${error.message} · 物理预演未接入`; }
}
$('scene-refresh').onclick = refreshScene;
async function loadSceneVersion() {
  try {
    const scene = await getJSON(`/v1/scene/versions/${$('scene-version').value}`);
    $('scene-data').textContent = JSON.stringify(scene, null, 2);
    $('scene-anchor').replaceChildren(...(scene.observation_ids || []).map((id) => {
      const option = document.createElement('option'); option.value = id; option.textContent = id; return option;
    }));
    $('scene-snapshot').disabled = readOnly() || !$('scene-anchor').value;
  }
  catch (error) { $('scene-data').textContent = `版本不可用 · ${error.message}`; }
}
$('scene-version').onchange = loadSceneVersion;
async function sceneOperation(path, value) {
  try {
    const result = await post(path, { request_id: crypto.randomUUID(), ...value });
    $('scene-data').textContent = JSON.stringify(result, null, 2);
    if (result.snapshot_id) $('scene-records').append(jsonDetails(`不可变几何快照 · ${result.snapshot_id}`, result));
    if (result.job_id) watchJob(result);
    return result;
  } catch (error) { $('scene-data').textContent = `内部操作未完成 · ${error.message}`; }
}
$('scene-query-form').onsubmit = (event) => {
  event.preventDefault();
  try { sceneOperation('/v1/scene/query', { ...JSON.parse($('scene-query').value), scene_version: Number($('scene-version').value) }); }
  catch (error) { $('scene-data').textContent = `查询参数无效 · ${error.message}`; }
};
$('scene-snapshot').onclick = () => sceneOperation('/v1/scene/snapshot', { scene_version: Number($('scene-version').value), anchor_observation_id: $('scene-anchor').value });
$('scene-align-form').onsubmit = (event) => {
  event.preventDefault();
  try { sceneOperation('/v1/scene/alignment', { ...JSON.parse($('scene-align').value), scene_version: Number($('scene-version').value) }); }
  catch (error) { $('scene-data').textContent = `对齐参数无效 · ${error.message}`; }
};
$('scene-update-form').onsubmit = (event) => {
  event.preventDefault();
  try { sceneOperation('/v1/scene/update', { ...JSON.parse($('scene-update').value), base_scene_version: Number($('scene-version').value) || 0 }); }
  catch (error) { $('scene-data').textContent = `几何更新参数无效 · ${error.message}`; }
};
async function watchJob(initial) {
  const label = initial.kind === 'build_scene' || initial.request?.base_scene_version !== undefined ? '内部几何更新' : '内部对齐';
  const details = jsonDetails(`${label} · ${initial.job_id} · ${initial.status}`, initial, true);
  $('scene-records').append(details);
  const cancel = document.createElement('button'); cancel.textContent = '取消该内部 job'; details.append(cancel);
  cancel.onclick = () => sceneOperation(`/v1/jobs/${encodeURIComponent(initial.job_id)}/cancel`, {});
  let job = initial;
  while (['queued', 'new', 'running'].includes(job.status)) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      job = await getJSON(`/v1/jobs/${encodeURIComponent(initial.job_id)}`);
      details.querySelector('summary').textContent = `${label} · ${job.job_id} · ${job.status}`;
      details.querySelector('pre').textContent = JSON.stringify(job, null, 2);
    } catch (error) { details.querySelector('pre').textContent = `job 状态未知 · ${error.message}`; break; }
  }
  cancel.disabled = true;
}
$('stop').onclick = async () => {
  $('stop').disabled = true;
  $('run-status').textContent = '正在请求停止 · 等待后端回执';
  try {
    const result = await getJSON('/v1/control', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(stopAttempt ||= { request_id: crypto.randomUUID(), command: 'stop' }),
    });
    $('run-status').textContent = result.status === 'stopped' ? '运行停止已确认' : '停止状态未确认 · 查看动作回执';
    $('run-data').textContent = JSON.stringify(result, null, 2);
    stopAttempt = undefined;
  } catch (error) { $('run-status').textContent = `停止请求失败 · ${error.message}`; }
  finally { $('stop').disabled = readOnly(); }
};

updateOwnership();
connectTerminal();
refreshRun();
refreshObservation();
refreshRecords();
refreshScene();
setInterval(() => { if (!historyMode()) refreshObservation(); }, 2000);
setInterval(refreshRun, 5000);
setInterval(() => { if (!historyMode()) refreshRecords(); }, 2000);
