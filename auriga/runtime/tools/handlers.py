"""Tool handlers. Registration binds these functions; dispatch has no name switch."""
from copy import deepcopy

from ...core.contracts import new_id
from ...core.tool_output import ToolOutput, ToolResponse, resolve_detail
from ...core.storage.images import image_url


def robot_information(runtime, args, correlation):
    if set(args) - {"section", "robot_id", "resource_id"}:
        raise ValueError("invalid_robot_information_arguments")
    section = args.get("section", "default")
    if not isinstance(section, str) or not section:
        raise ValueError("invalid_robot_information_section")
    robot_id = args.get("robot_id")
    if robot_id is not None and (not isinstance(robot_id, str) or not robot_id):
        raise ValueError("invalid_robot_information_robot_id")
    if "resource_id" in args and section != "resource":
        raise ValueError("robot_resource_id_requires_resource_section")
    if section in {"resources", "resource"}:
        from ...robot.model.robot_information import robot_resources, robot_resource_path
        selected = robot_id if robot_id is not None else (runtime.capabilities or {}).get("robot_model_id")
        result = robot_resources(selected or "unknown")
        if section == "resource":
            resource_id = args.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("robot_resource_id_required")
            path = robot_resource_path(selected or "unknown", resource_id)
            if path is None:
                result = dict(robot_id=selected, available=False, error="robot_resource_unavailable")
            else:
                resource = next(item for item in result["resources"] if item["resource_id"] == resource_id)
                result = dict(robot_id=selected, available=True, resource=resource,
                    resource_ref=runtime.store.put_artifact(path.read_bytes(),
                        "robot_geometry_model" if resource["format"] == "ZIP" else "robot_cad",
                        "application/zip" if resource["format"] == "ZIP" else "model/step"))
        return ToolOutput(result, projection="owner_record").response()
    if section == "geometry_image":
        from ...robot.model.robot_information import robot_geometry_path
        selected = robot_id if robot_id is not None else (runtime.capabilities or {}).get("robot_model_id")
        path = robot_geometry_path(selected or "unknown")
        if path is None:
            return ToolOutput(dict(robot_id=selected, available=False, section=section,
                                   error="robot_geometry_image_unavailable")).response()
        url, _ = image_url(path.read_bytes(), runtime.image_settings)
        return ToolOutput(dict(robot_id=selected, available=True, section=section,
            description="Illustrative robot geometry and contact references; not a measured scene or executed trajectory."),
            extra_content=(dict(type="inputImage", imageUrl=url),)).response()
    return ToolOutput(runtime.robot_information(section, robot_id), projection="robot_information").response()


def robot_observe(runtime, args, correlation):
    camera_id, roi, scale = args.get('camera_id'), args.get('roi'), args.get('scale', 1)
    if (roi is not None or scale != 1) and camera_id is None:
        raise ValueError("image_view_requires_camera_id")
    if type(scale) is not int or not 1 <= scale <= 4:
        raise ValueError("invalid_image_scale")
    if camera_id is not None and runtime.latest is not None and not runtime.active_action:
        bundle = {**runtime.latest, "action_budget": runtime.action_budget()}
    else:
        bundle = runtime.observe(correlation)
    content = runtime.observation_content(bundle, correlation, camera_id, roi, scale,
                                          detail=args.get('detail'), structured=True)
    if camera_id is None:
        from ...perception.external.obs3d import attachments as obs3d_attachments
        content = [*content, *obs3d_attachments(runtime, bundle, correlation)]
    if correlation.get('program_id'):
        obs_val = deepcopy(content[0].value) if content and hasattr(content[0], 'value') else {}
        obs_data = {**(obs_val if isinstance(obs_val, dict) else {}), **bundle}
    else:
        obs_data = bundle
    return ToolResponse(content, data=obs_data, kind='observation', data_projection='observation')


def history_view_observation(runtime, args, correlation):
    content = runtime.observation_content(runtime.store.read("observation", args["observation_id"]), correlation,
                                       args.get("camera_id"), args.get("roi"), args.get("scale", 1), detail=args.get("detail"), structured=True)
    obs_val = deepcopy(content[0].value)
    obs_data = {**(obs_val.get("bundle") if isinstance(obs_val, dict) and isinstance(obs_val.get("bundle"), dict) else {}),
                **(obs_val if isinstance(obs_val, dict) else {})}
    return ToolResponse(content, data=obs_data)


def task_plan(runtime, args, correlation):
    with runtime.lock:
        result = runtime.task_planning.call(args, runtime.identity, runtime.latest, correlation)
    return ToolOutput(result, projection="owner_record").response()


def robot_capabilities(runtime, args, correlation):
    detail = resolve_detail(args.get("detail"), program=bool(correlation.get("program_id")),
        default="full" if runtime.actor_payload_profile == "full_v1" else "summary")
    result = (runtime._capability_delivery.project(runtime.capabilities,
        epoch=runtime._model_context_epoch(), current_action_budget=runtime.action_budget())
        if detail == "summary" and not correlation.get("program_id") else runtime.capabilities)
    return ToolOutput(result, projection="capabilities").response()


def vision_compare(runtime, args, correlation):
    from ...perception.visual_feedback import compare_observations
    result = compare_observations(runtime.store, runtime.identity, **args)
    return ToolOutput(result, projection="owner_record").response()


def robot_action_status(runtime, args, correlation):
    from ...core.presentation.history_output import action_status
    result = action_status(runtime, args, correlation)
    return ToolOutput(result, projection="action_result").response()


def history_list_actions(runtime, args, correlation):
    from ...core.presentation.history_output import list_actions
    with runtime.lock:
        result = list_actions(runtime, args, correlation)
    return ToolOutput(result, projection="action_history").response()


def history_read_trace(runtime, args, correlation):
    from ...core.presentation.history_output import read_trace
    result = read_trace(runtime, args, correlation)
    return ToolOutput(result).response()


def history_annotate_action(runtime, args, correlation):
    runtime.action(args["action_id"])
    if not isinstance(args["note"], str) or len(args["note"]) > 16000: raise ValueError("invalid_action_note")
    result = runtime.store.save("action_note", new_id("note"), dict(action_id=args["action_id"], note=args["note"], source_kind="actor_judgment", correlation=correlation))
    return ToolOutput(result, projection="owner_record").response()


def program_status(runtime, args, correlation):
    from ..programs.program_result import program_result
    from .program_tools import program_content
    detail = resolve_detail(args.get("detail"), program=bool(correlation.get("program_id")))
    job = runtime.programs.status(args["job_id"])
    result = program_result(runtime.store, job, detail=detail)
    return ToolResponse(program_content(runtime, job, result, detail, correlation), data=result, data_projection='owner_record')


def vision_review(runtime, args, correlation):
    args = dict(args)
    operation = args.pop('evidence_mode', 'images') == 'operation'
    reviewer = runtime.visual_reviewer.review_operation if operation else runtime.visual_reviewer.review
    return ToolOutput(reviewer(runtime.store, runtime.identity, **args,
        **runtime._review_controls(correlation)),
        projection="operation_review" if operation else "owner_record").response()


def robot_contact_overlay(runtime, args, correlation):
    from ..actions.inspection import contact_overlay
    content = contact_overlay(runtime, args, correlation, structured=True)
    return ToolResponse(content, data=deepcopy(content[0].value), data_projection="contact_overlay")


def robot_action(runtime, args, correlation):
    from ..actions.unified_actions import dispatch
    return dispatch(runtime, args, correlation)


def robot_wait(runtime, args, correlation):
    from .actions import execute_action
    return execute_action(runtime, 'robot_wait', args, correlation)


def robot_policy_step(runtime, args, correlation):
    from .actions import execute_action
    return execute_action(runtime, 'robot_policy_step', args, correlation)


def robot_policy_delegate(runtime, args, correlation):
    from ..policy.delegation import delegation_response
    return delegation_response(runtime, args, correlation)


def program_run(runtime, args, correlation):
    from .program_tools import run_program
    return run_program(runtime, 'program_run', args, correlation)


def operation_memory(runtime, args, correlation):
    return ToolOutput(runtime.operation_memory.call(args['request'], correlation)).response()


def depth_query(runtime, args, correlation):
    result = runtime.depth.query(args['request'], correlation)
    if not correlation.get('program_id'):
        from ...perception.depth.depth_payload import numeric_summary
        result = numeric_summary(result)
    return ToolOutput(result, projection="owner_record").response()


def object_table_query(runtime, args, correlation):
    """features.object_table: geometry_query kind object_table (auriga/object_table.py)."""
    from ...perception.external.object_table import object_table_query as run_object_table
    result, overlays = run_object_table(runtime, args['request'])
    val = result.get("value")
    obs_id = args["request"].get("observation_id")
    if isinstance(val, dict) and obs_id:
        from ...perception.external.obs3d import _remember
        if (obs_id, "object_table") not in getattr(runtime, "_obs3d_cache", {}):
            shown = dict(columns=val.get("columns"), rows=val.get("rows"), table_z=val.get("support_z_m"))
            if val.get("grasp_hints"):
                shown["grasp_hints"] = val["grasp_hints"]
            if val.get("held"):
                shown["held"] = val["held"]
            _remember(runtime, obs_id, "object_table", (shown, None, result))
    model_copy = {key: value for key, value in result.items() if key != "diagnostics"}
    from ...core.storage.images import image_settings
    settings = runtime.image_settings if runtime.image_settings is not None else image_settings(None)
    content = ToolOutput(model_copy, projection="geometry_result").content_parts()
    for camera_id, data, original in overlays:
        reference = runtime.store.put_artifact(data, "geometry_overlay", "image/jpeg")
        url, encoding = image_url(data, settings)
        content.append(dict(type="inputImage", imageUrl=url))
        runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": encoding["sha256"],
            "sha256": encoding["sha256"], "encoding": encoding, "original_rgb_ref": original, "camera_id": camera_id,
            "source_kind": "object_table_overlay", "delivered_to_model": not bool(correlation.get("program_id"))},
            correlation, world_kind="internal")
        result.setdefault("overlay_refs", {})[camera_id] = reference
    return ToolResponse(content, data=result, data_projection='geometry_result')


def scene3d_query(runtime, args, correlation):
    """features.scene3d: geometry_query kind scene3d (auriga/scene3d.py); the cloud is stored, not returned."""
    from ...perception.external.scene3d import scene3d_query as run_scene3d, ARTIFACT_KIND, ARTIFACT_MIME
    result, data = run_scene3d(runtime, args['request'])
    if data is not None:
        reference = runtime.store.put_artifact(data, ARTIFACT_KIND, ARTIFACT_MIME)
        result["value"]["artifact"] = reference
        runtime.store.event("geometry.scene3d_stored", {"artifact": reference, "bytes": len(data),
            "points": result["value"]["points"], "observation_id": args['request'].get("observation_id"),
            "camera_id": result["value"]["camera_id"]}, correlation, world_kind="internal")
    model_copy = {key: value for key, value in result.items() if key != "diagnostics"}
    return ToolResponse(ToolOutput(model_copy, projection="geometry_result").content_parts(), data=result,
                        data_projection='geometry_result')


def geometry_query(runtime, args, correlation):
    req = args['request']
    if isinstance(req.get('queries'), list) and req.get('kind') in (None, 'batch'):
        batch_results, extra_images = [], []
        _kind_aliases = {'ray': 'pixel_ray', 'project': 'project_points'}
        for item in req['queries']:
            sub_req = dict(item) if isinstance(item, dict) else {}
            for inherited in (
                'observation_id', 'camera_id', 'artifact', 'plane_origin_m',
                'plane_normal', 'plane_source', 'plane_origin_alt_m', 'target_camera_id',
            ):
                if inherited in req and inherited not in sub_req:
                    sub_req[inherited] = req[inherited]
            if sub_req.get('kind') in _kind_aliases:
                sub_req['kind'] = _kind_aliases[sub_req['kind']]
            if 'op' in sub_req and 'kind' not in sub_req:
                sub_req = {'kind': 'scene3d', 'query': dict(item),
                           **{k: sub_req[k] for k in ('observation_id', 'camera_id', 'artifact') if k in sub_req}}
            sub_resp = geometry_query(runtime, {'request': sub_req}, correlation)
            sub_data = sub_resp.data if isinstance(sub_resp.data, dict) else {}
            batch_results.append({k: sub_data[k] for k in ('query_type', 'status', 'reason', 'value', 'assumptions')
                                  if k in sub_data})
            for part in sub_resp.content:
                if isinstance(part, dict) and part.get('type') in ('inputImage', 'input_image', 'image'):
                    extra_images.append(part)
        obs_ids = list(dict.fromkeys(
            ([req['observation_id']] if req.get('observation_id') else [])
            + [q.get('observation_id') for q in req['queries'] if isinstance(q, dict) and q.get('observation_id')]
        ))
        common_assumptions = []
        if len(batch_results) > 1 and all('assumptions' in r for r in batch_results):
            first_a = batch_results[0]['assumptions']
            if all(r['assumptions'] == first_a for r in batch_results):
                common_assumptions = first_a
                for r in batch_results:
                    r.pop('assumptions', None)
        result = {
            'schema_version': '0.1',
            **runtime.identity,
            'world_kind': 'internal',
            'query_type': 'batch',
            'observation_ids': obs_ids,
            'status': 'supported',
            'frame_id': 'env_origin',
            'unit': 'm',
            **({'assumptions': common_assumptions} if common_assumptions else {}),
            'value': {'count': len(batch_results), 'batch_results': batch_results},
        }
        content = [*ToolOutput(result, projection='geometry_result').content_parts(), *extra_images]
        return ToolResponse(content, data=result, data_projection='geometry_result')
    from ...perception.depth.depth_runtime import DEPTH_KINDS
    if req.get('kind') in DEPTH_KINDS:
        return depth_query(runtime, args, correlation)
    from ...robot.model.geometry import GeometryTools
    from ...perception.external.object_table import applies as object_table_applies
    if object_table_applies(runtime, req):
        return object_table_query(runtime, args, correlation)
    from ...perception.external.scene3d import applies as scene3d_applies
    if scene3d_applies(runtime, req) or (req.get('kind') == 'scene3d' and ('artifact' in req or 'query' in req or 'queries' in req)):
        return scene3d_query(runtime, args, correlation)
    from ...perception.external.vggt_height import applies as vggt_height_applies
    if vggt_height_applies(runtime, req):
        from ...perception.external.vggt_height import pixel_plane_with_estimate
        result = pixel_plane_with_estimate(runtime, req)
    else:
        result = GeometryTools(runtime.scene, settings=runtime.image_settings).execute(
            {**req, **runtime.identity, 'schema_version': '0.1'})
    content = ToolOutput(result, projection="geometry_result").content_parts()
    if isinstance(result.get('value'), dict) and result['value'].get('overlay_ref'):
        reference = result["value"]["overlay_ref"]
        info, data = runtime.scene.resolve_artifact(reference)
        copied = runtime.store.put_artifact(data, info["kind"], info["mime_type"], info["access_scope"])
        if copied != reference: raise ValueError("geometry_overlay_copy_identity_mismatch")
        url, encoding = image_url(data, runtime.image_settings)
        content.append(dict(type="inputImage", imageUrl=url))
        runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": info["sha256"],
            "sha256": encoding["sha256"], "encoding": encoding,
            "original_rgb_ref": result["value"].get("original_rgb_ref"), "source_kind": "geometry_overlay",
            "delivered_to_model": not bool(correlation.get("program_id"))}, correlation, world_kind="internal")
    return ToolResponse(content, data=result, data_projection='geometry_result')


def pose_plan(runtime, args, correlation):
    """features.pose_skill: read-only pose planning from the object table (auriga/pose_skill.py)."""
    from ...perception.external.pose_skill import handle
    return handle(runtime, args, correlation)
