"""HTTP, SSE and WebSocket routes of one workbench."""

import asyncio
import base64
import json
import hashlib

from aiohttp import web, WSMsgType
from openai_codex.errors import CodexError, JsonRpcError, ServerBusyError

from ...core.contracts import canonical, digest, IDENTITY
from ...cortex.agents.base import AgentRejected, AgentTransportUnknown
from ...core.storage.store import file_digest
from ..library.task_review import TaskReviewService
from ...core.storage.read_models import artifact_response, record_updates
from ..library.static_assets import static_directory
from ...perception.scene.scene_service import TERMINAL as JOB_TERMINAL
from ..library.experiments import catalog_id, contained
from ..evaluation.episode import record_video


class WorkbenchRoutes:
    def app(self):
        @web.middleware
        async def errors(request, handler):
            try:
                if request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
                    raise web.HTTPForbidden(text="loopback_host_required")
                if request.headers.get("X-Auriga-Run-ID", self.identity["run_id"]) != self.identity["run_id"]:
                    raise web.HTTPNotFound(text="run_identity_mismatch")
                if request.match_info.get("run_id", self.identity["run_id"]) != self.identity["run_id"]:
                    raise web.HTTPNotFound(text="run_identity_mismatch")
                if request.method == "POST" and request.headers.get("Origin") not in (None, f"http://{request.host}"):
                    raise web.HTTPForbidden(text="same_origin_required")
                return await handler(request)
            except FileNotFoundError:
                return web.json_response({"error": "artifact_original_unavailable"}, status=404)
            except KeyError as error:
                return web.json_response({"error": f"missing:{error}"}, status=404)
            except (ValueError, TypeError) as error:
                return web.json_response({"error": str(error)}, status=409 if "conflict" in str(error) else 400)

        app = web.Application(middlewares=[errors], client_max_size=1024 * 1024)
        task_reviews = TaskReviewService(self.config.library.task_source_root, config=self.config)
        # A standalone viewer owns exactly one run; no library scan or actor tool is involved.
        review_id = catalog_id(".")

        async def task_review(request):
            entry = dict(path=self.root, metadata={}, item=dict(id=review_id, task=self.task, status=self.status))
            return web.json_response(await asyncio.to_thread(task_reviews.get, entry))

        async def review_file(request):
            if request.match_info["id"] != review_id:
                raise web.HTTPNotFound(text="unindexed_evidence_file")
            # These are the only downloads linked by TaskReviewService. Never expose
            # manifest credentials, private files, task source, or arbitrary paths.
            for name in ("summary.json", "evaluation.json"):
                if request.match_info["asset"] == hashlib.sha256(name.encode()).hexdigest()[:32]:
                    path = self.root / name
                    if path.is_symlink() or not path.is_file():
                        raise web.HTTPNotFound(text="unindexed_evidence_file")
                    contained(self.root, name)
                    sha = await asyncio.to_thread(file_digest, path)
                    return web.FileResponse(path, headers={"Content-Type": "application/json", "X-Evidence-SHA256": sha})
            raise web.HTTPNotFound(text="unindexed_evidence_file")

        async def run(request): return web.json_response(self.snapshot())

        async def observation(request):
            identifier = request.match_info.get("id")
            return web.json_response(self.store.observation_view(self.harness.latest if identifier == "latest" else self.store.read("observation", identifier)))

        async def artifact(request):
            reference="artifact:"+request.match_info["id"]
            try:
                return await artifact_response(self.store, reference, image=request.query.get("view") == "image")
            except KeyError:
                if not self.scene_service: raise
                return await artifact_response(self.scene_service.store, reference, allowed_scopes=("actor", "internal"))

        async def action(request): return web.json_response(self.harness.action(request.match_info["id"]))

        async def events(request):
            after = max(int(request.query.get("after", "0")), int(request.headers.get("Last-Event-ID", "0")))
            if after < 0: raise ValueError("invalid_event_cursor")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            if after > self.store.event_cursor():
                after=self.store.event_cursor()
                await response.write(f'event: gap\ndata: {canonical({"reason":"cursor_ahead_or_run_reset","current_seq":after})}\n\n'.encode())
            try:
                while True:
                    batch = self.store.events(after)
                    for event in batch:
                        await response.write(f'id: {event["server_seq"]}\ndata: {canonical(event)}\n\n'.encode())
                        after = event["server_seq"]
                    if not batch:
                        await response.write(b": live\n\n")
                        await asyncio.sleep(0.2)
            except (ConnectionResetError, asyncio.CancelledError):
                pass
            return response

        async def terminal(request):
            if not self.terminal: raise web.HTTPServiceUnavailable(text="terminal_not_started")
            if request.headers.get("Origin") not in (None, f"http://{request.host}"):
                raise web.HTTPForbidden(text="same_origin_required")
            ws = web.WebSocketResponse(max_msg_size=4096)
            await ws.prepare(request)
            after = int(request.query.get("after", "0"))
            prior_session=request.query.get("session_id")
            if prior_session and prior_session!=self.terminal.session_id:
                await ws.send_json(dict(type="gap",reason="terminal_session_changed",current_cursor=self.terminal.cursor))
                after=0
            await ws.send_json(dict(type="ready", status=self.terminal.status()))
            if after<0 or after>self.terminal.cursor:
                await ws.send_json(dict(type="gap",reason="terminal_cursor_outside_retained_stream",current_cursor=self.terminal.cursor))
                after=0

            async def output():
                async for data, cursor in self.terminal.stream(after):
                    await ws.send_json(dict(type="output", cursor=cursor, after=cursor-len(data), data=base64.b64encode(data).decode()))
                await ws.send_json(dict(type="status", status=self.terminal.status()))
            task = asyncio.create_task(output())
            try:
                async for message in ws:
                    if message.type != WSMsgType.TEXT: continue
                    try:
                        value = json.loads(message.data)
                        if value.get("type") != "resize": raise ValueError("terminal_input_is_read_only")
                        self.terminal.resize(value["cols"], value["rows"])
                    except (ValueError, KeyError) as error:
                        await ws.send_json(dict(type="error", message=str(error)))
            finally:
                task.cancel(); await asyncio.gather(task, return_exceptions=True)
            return ws

        async def control(request):
            value=await request.json()
            if not isinstance(value,dict) or not isinstance(value.get("request_id"),str) or not 0<len(value["request_id"])<=256: raise ValueError("control_requires_request_id")
            identifier=value["request_id"]
            async with self.control_lock:
                self.store.save("control_request",identifier,value)
                saved=self.store.read_state("control_response",identifier)
                if saved:
                    return web.json_response(saved, status=409 if saved.get("status")=="unknown" else saved.get("http_status",200))
                # Persist admission before external dispatch. Crash ambiguity never resends a turn.
                self.store.set_state("control_response",identifier,{"request_id":identifier,"status":"unknown","reason":"dispatch_acknowledgement_missing_do_not_resend"})
                try:
                    result=await self.execute_control(value,request.headers.get("Origin"))
                except (JsonRpcError, AgentRejected) as error:
                    uncertain = self.session.state in ("running", "disconnected", "unknown")
                    if uncertain:
                        self.status = "unknown"
                    result = dict(request_id=identifier, status="unknown" if uncertain else "rejected",
                        reason=getattr(error, "message", str(error)), native_rpc_code=getattr(error, "code", None),
                        http_status=409 if uncertain else 429 if (isinstance(error, ServerBusyError) or getattr(error, "busy", False)) else 400)
                    self.store.event("control.status_unknown" if uncertain else "control.rejected", result)
                except (CodexError, AgentTransportUnknown) as error:
                    self.status = "unknown"
                    result = dict(request_id=identifier, status="unknown", reason=str(error), http_status=409)
                    self.store.event("control.status_unknown", result)
                except (RuntimeError,TimeoutError,OSError) as error:
                    self.status="unknown"
                    result={"request_id":identifier,"status":"unknown","reason":str(error),"http_status":409}
                    self.store.event("control.status_unknown",result)
                except (ValueError,KeyError) as error:
                    result={"request_id":identifier,"error":str(error),"http_status":409 if "conflict" in str(error) else 400}
                self.store.set_state("control_response",identifier,result)
                return web.json_response(result,status=result.get("http_status",200))

        async def observations(request): return web.json_response({"observations":[self.store.observation_view(o) for o in self.store.records("observation")]})
        async def records(request): return web.json_response(record_updates(self.store, request.query))
        async def items(request):
            identifier=request.match_info.get("id")
            if identifier:
                item=self.store.read_state("codex_item",identifier)
                if item is None: raise KeyError(identifier)
                return web.json_response(item)
            return web.json_response({"items":self.store.states("codex_item"),"cursor":self.store.event_cursor()})
        async def requests(request): return web.json_response({"requests":self.store.states("native_request")})
        async def scene_get(request):
            kind=request.match_info["kind"]
            if request.match_info.get("version"): result=self.harness.scene.version(self.identity,int(request.match_info["version"]))
            elif kind=="head": result=self.harness.scene.head(self.identity)
            elif kind=="versions": result={"versions":self.harness.scene.versions(self.identity)}
            else: result=self.harness.scene.version(self.identity,int(request.match_info["version"]))
            return web.json_response(result)
        async def scene_post(request):
            if self.replay: raise ValueError("replay_is_read_only")
            if not self.scene_service: raise ValueError("scene_service_not_started")
            value=await request.json()
            if not isinstance(value,dict): raise ValueError("scene_request_requires_object")
            if any(key in value and value[key]!=self.identity[key] for key in IDENTITY): raise ValueError("scene_identity_conflict")
            identifier=value.get("request_id") or request.headers.get("Idempotency-Key")
            if not isinstance(identifier,str) or not 0<len(identifier)<=256: raise ValueError("scene_requires_request_id")
            scope="scene-"+digest([request.path,identifier])
            self.store.save("scene_request",scope,value)
            old=self.store.read_state("scene_response",scope)
            if old: return web.json_response(old)
            body={key: item for key, item in value.items() if key not in {"request_id", "request"}}
            body.update(self.identity, schema_version="0.1")
            operation=request.match_info["operation"]
            if operation=="query": result=self.harness.scene.query(self.identity,value["scene_version"],value.get("request",value))
            elif operation=="snapshot": result=self.harness.scene.snapshot(self.identity,value["scene_version"],value["anchor_observation_id"])
            else:
                body.update(value.get("request",{})); body.update(self.identity)
                result=self.scene_service.submit(body,"alignment" if operation=="alignment" else "build_scene",identifier="job-"+scope)
            self.store.set_state("scene_response",scope,result)
            return web.json_response(result)
        async def job(request):
            if self.replay and request.method=="POST": raise ValueError("replay_is_read_only")
            identifier=request.match_info["id"]
            if identifier.startswith("program-"):
                if self.replay:
                    result=self.store.read_state("program",identifier)
                    if result is None: raise KeyError(identifier)
                else: result=self.harness.programs.cancel(identifier) if request.method=="POST" else self.harness.programs.status(identifier)
            else:
                result=self.scene_service.job(identifier)
                if any(result[key]!=self.identity[key] for key in IDENTITY): raise ValueError("job_identity_conflict")
                if request.method=="POST" and result["status"] not in JOB_TERMINAL: result=self.scene_service.transition(result,result["status"],cancel_requested=True,cancel_reason="web_request")
            return web.json_response(result)
        async def evaluate(request):
            return web.json_response(await asyncio.to_thread(self.observer_evaluate))

        async def observer_snapshot(request):
            def capture():
                with self.harness.robot_lock:
                    self.require_observer_idle()
                    result=self.robot.observer_snapshot()
                    evidence={**self.identity,"audience":"observer_only","snapshot":result}
                    evidence["snapshot_ref"]=self.store.put_artifact(canonical(evidence).encode(),"observer_scene_snapshot","application/json","observer")
                    return evidence
            return web.json_response(await asyncio.to_thread(capture))

        async def record_video(request):
            value=await request.json()
            if type(value.get("enabled",True)) is not bool: raise ValueError("invalid_recording_enabled")
            def record():
                with self.harness.robot_lock:
                    self.require_observer_idle()
                    result={**self.identity,**self.robot.record(enabled=value.get("enabled",True))}
                    self.store.event("observer.recording",result,source="private_evaluator")
                    return result
            return web.json_response(await asyncio.to_thread(record))

        async def video(request):
            return web.json_response(await asyncio.to_thread(self.observer_video))

        def observer_guard(handler):
            async def guarded(request):
                async with self.control_lock:
                    self.require_observer_idle()
                    return await handler(request)
            return guarded
        evaluate=observer_guard(evaluate)
        observer_snapshot=observer_guard(observer_snapshot)
        record_video=observer_guard(record_video)
        video=observer_guard(video)

        app.router.add_get("/v1/observations",observations)
        app.router.add_get("/v1/task-review",task_review)
        app.router.add_get("/v1/experiments/{id}/files/{asset}",review_file)
        app.router.add_get("/v1/records",records)
        app.router.add_get("/v1/items",items)
        app.router.add_get("/v1/items/{id}",items)
        app.router.add_get("/v1/requests",requests)
        app.router.add_get("/v1/scene/{kind:head|versions}",scene_get)
        app.router.add_get("/v1/scene/{kind:versions}/{version}",scene_get)
        app.router.add_post("/v1/scene/{operation:query|snapshot|alignment|update}",scene_post)
        app.router.add_get("/v1/jobs/{id}",job)
        app.router.add_post("/v1/jobs/{id}/cancel",job)
        app.router.add_get("/v1/observer/snapshot",observer_snapshot)
        app.router.add_post("/v1/observer/evaluate",evaluate)
        app.router.add_post("/v1/observer/video",video)
        app.router.add_post("/v1/observer/record",record_video)
        app.router.add_get("/v1/run", run)
        app.router.add_get("/v1/runs/{run_id}", run)
        app.router.add_get("/v1/events", events)
        app.router.add_get("/v1/runs/{run_id}/events", events)
        app.router.add_get("/v1/observations/{id}", observation)
        app.router.add_get("/v1/artifacts/{id}", artifact)
        app.router.add_get("/v1/actions/{id}", action)
        app.router.add_get("/v1/terminal", terminal)
        app.router.add_post("/v1/control", control)
        app.router.add_post("/v1/runs/{run_id}/control", control)
        static = static_directory()
        async def index(request): return web.FileResponse(static / "index.html")
        app.router.add_get("/", index)
        if static.exists(): app.router.add_static("/", static)
        return app
