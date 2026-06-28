from .reports import feature_set, response_model_name, status_suffix


def render_example(
    *,
    entity: dict,
    blueprint: dict,
    task_type: str,
    slm_id: str,
    variant_number: int,
) -> dict:
    function_name = function_name_for(entity, blueprint, variant_number)
    route_path = route_path_for(entity, blueprint)
    prompt = render_prompt(entity, blueprint, function_name, route_path)
    target = render_target(entity, blueprint, function_name, route_path, variant_number)
    return {
        "id": f"{slm_id}-{blueprint['template_id']}-{variant_number}",
        "task_type": task_type,
        "prompt": prompt,
        "target": target,
        "metadata": {"template_id": blueprint["template_id"], "features": feature_set(blueprint)},
    }


def function_name_for(entity: dict, blueprint: dict, variant_number: int) -> str:
    if blueprint["method"] == "POST":
        return f"create_{entity['singular']}_{variant_number}"
    if blueprint["response_kind"] == "list":
        return f"list_{entity['plural']}_{variant_number}"
    return f"get_{entity['singular']}_{variant_number}"


def route_path_for(entity: dict, blueprint: dict) -> str:
    path = f"/{entity['plural']}"
    if blueprint["include_path"]:
        path += f"/{{{entity['singular']}_id}}"
    if blueprint["method"] == "GET" and blueprint["response_kind"] == "list" and blueprint["include_auth"]:
        path += "/audit"
    if blueprint.get("include_nested_body"):
        path += "/bulk"
    return path


def render_prompt(entity: dict, blueprint: dict, function_name: str, route_path: str) -> str:
    parts = [
        f"Write FastAPI code for a {blueprint['method']} endpoint named {function_name} mounted at {route_path}.",
        f"Use response_model {response_model_name(entity, blueprint)} and status code {blueprint['status_code']}.",
    ]
    if blueprint["include_path"]:
        parts.append(f"Validate the path parameter {entity['singular']}_id with Path(..., ge=1).")
    if blueprint["include_query"]:
        parts.append(f"Accept query params {entity['query_flag']}: bool and max_{entity['plural']}: int with Query validation.")
    if blueprint["include_body"]:
        parts.append(f"Use request body model {entity['request_model']} with Pydantic Field validation.")
    if blueprint.get("include_nested_body"):
        parts.append("Include a nested Pydantic model inside the request body.")
    if blueprint["include_dependency"]:
        parts.append("Inject db with Depends(get_db).")
    if blueprint["include_auth"]:
        parts.append("Also inject current_user with Depends(require_auth).")
    parts.append(f"Raise HTTPException with status code {blueprint['error_status']} on the main failure path. Return code only.")
    return " ".join(parts)


def render_target(entity: dict, blueprint: dict, function_name: str, route_path: str, variant_number: int) -> str:
    lines = [
        "from fastapi import APIRouter, Depends, HTTPException, Path, Query, status",
        "from pydantic import BaseModel, Field",
        "",
        "router = APIRouter()",
        "",
        "",
        "def get_db() -> dict[str, str]:",
        '    return {"session": "primary"}',
    ]
    if blueprint["include_auth"]:
        lines.extend(["", "", "def require_auth() -> dict[str, str]:", '    return {"role": "service"}'])
    if blueprint.get("include_nested_body"):
        lines.extend(["", "", f"class {entity['request_model']}Details(BaseModel):", "    region: str = Field(..., min_length=2, max_length=20)", "    priority: int = Field(..., ge=1, le=5)"])
    if blueprint["include_body"]:
        first_name, first_type, first_validator = entity["first_field"]
        second_name, second_type, second_validator = entity["second_field"]
        lines.extend(["", "", f"class {entity['request_model']}(BaseModel):", f"    {first_name}: {first_type} = {first_validator}", f"    {second_name}: {second_type} = {second_validator}"])
        if blueprint.get("include_nested_body"):
            lines.append(f"    details: {entity['request_model']}Details")
    response_model = response_model_name(entity, blueprint)
    first_name, first_type, first_validator = entity["first_field"]
    status_field = entity["status_field"]
    status_active = entity["status_values"][0]
    status_secondary = entity["status_values"][1]
    lines.extend(["", "", f"class {response_model}(BaseModel):", "    id: int = Field(..., ge=1)", f"    {first_name}: {first_type} = {first_validator}", f"    {status_field}: str"])
    lines.extend(["", "", f"@router.{blueprint['method'].lower()}(", f'    "{route_path}",', f"    response_model={response_model}," if blueprint["response_kind"] == "single" else f"    response_model=list[{response_model}],", f"    status_code=status.HTTP_{status_suffix(blueprint['status_code'])},", ")"])
    params = []
    if blueprint["include_path"]:
        params.append(f"{entity['singular']}_id: int = Path(..., ge=1, description=\"{entity['singular'].title()} identifier\")")
    if blueprint["include_query"]:
        params.append(f"{entity['query_flag']}: bool = Query(False, description=\"Include related objects\")")
        params.append(f"max_{entity['plural']}: int = Query(25, ge=1, le=100, description=\"Result size\")")
    if blueprint["include_body"]:
        params.append(f"payload: {entity['request_model']}")
    params.append("db: dict[str, str] = Depends(get_db)")
    if blueprint["include_auth"]:
        params.append("current_user: dict[str, str] = Depends(require_auth)")
    return_type = response_model if blueprint["response_kind"] == "single" else f"list[{response_model}]"
    lines.append(f"def {function_name}(")
    for param in params:
        lines.append(f"    {param},")
    lines.append(f") -> {return_type}:")
    if blueprint["include_auth"]:
        lines.append('    if current_user.get("role") == "blocked":')
        lines.append(f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="access denied")')
    if blueprint["include_path"]:
        lines.append(f"    if {entity['singular']}_id == 9999:")
        lines.append(f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="{entity["missing_detail"]}")')
    elif blueprint["include_body"]:
        lines.append(f'    if payload.{entity["first_field"][0]} == "{entity["duplicate_value"]}":')
        lines.append(f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="{entity["duplicate_detail"]}")')
    else:
        lines.append(f"    if max_{entity['plural']} < 1:")
        lines.append(f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="invalid query")')
    identity_value = f"{entity['singular']}_id" if blueprint["include_path"] else str(variant_number)
    primary_value = f"payload.{entity['first_field'][0]}" if blueprint["include_body"] else f'"{entity["singular"]}-{variant_number}"'
    if blueprint["response_kind"] == "single":
        lines.append(f"    return {response_model}(id={identity_value}, {entity['first_field'][0]}={primary_value}, {status_field}=\"{status_active}\")")
    else:
        lines.append("    return [")
        lines.append(f"        {response_model}(id={variant_number}, {entity['first_field'][0]}={primary_value}, {status_field}=\"{status_active}\"),")
        lines.append(f"        {response_model}(id={variant_number + 1}, {entity['first_field'][0]}=\"{entity['singular']}-{variant_number + 1}\", {status_field}=\"{status_secondary}\"),")
        lines.append("    ]")
    return "\n".join(lines) + "\n"
