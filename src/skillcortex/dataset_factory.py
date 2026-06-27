import json
import random
from pathlib import Path

from .datasets import default_report_path, validate_training_datasets, write_validation_report


DEFAULT_DATASET_SEED = 42
DEFAULT_EVAL_RATIO = 0.2
SUPPORTED_DOMAINS = {"fastapi": "fastapi_contract", "fastapi_contract": "fastapi_contract"}
REQUIRED_FASTAPI_FEATURES = [
    "get_endpoint",
    "post_endpoint",
    "path_params",
    "query_params",
    "request_body",
    "response_model",
    "error_handling",
    "dependency_injection",
    "status_codes",
    "pydantic_validation",
]

ENTITY_VARIANTS = (
    {
        "singular": "user",
        "plural": "users",
        "request_model": "CreateUserRequest",
        "response_model": "UserResponse",
        "list_model": "UserSummary",
        "first_field": ("name", "str", "Field(..., min_length=3, max_length=40)"),
        "second_field": ("email", "str", "Field(..., min_length=5, max_length=80)"),
        "status_field": "status",
        "status_values": ("active", "pending"),
        "query_flag": "include_orders",
        "duplicate_value": "duplicate-user",
        "missing_detail": "user not found",
        "duplicate_detail": "user already exists",
    },
    {
        "singular": "order",
        "plural": "orders",
        "request_model": "CreateOrderRequest",
        "response_model": "OrderResponse",
        "list_model": "OrderSummary",
        "first_field": ("reference", "str", "Field(..., min_length=4, max_length=24)"),
        "second_field": ("quantity", "int", "Field(..., ge=1, le=50)"),
        "status_field": "state",
        "status_values": ("queued", "confirmed"),
        "query_flag": "include_items",
        "duplicate_value": "duplicate-order",
        "missing_detail": "order not found",
        "duplicate_detail": "order already exists",
    },
    {
        "singular": "invoice",
        "plural": "invoices",
        "request_model": "CreateInvoiceRequest",
        "response_model": "InvoiceResponse",
        "list_model": "InvoiceSummary",
        "first_field": ("code", "str", "Field(..., min_length=3, max_length=16)"),
        "second_field": ("amount_cents", "int", "Field(..., ge=100, le=500000)"),
        "status_field": "status",
        "status_values": ("draft", "paid"),
        "query_flag": "include_receipts",
        "duplicate_value": "duplicate-invoice",
        "missing_detail": "invoice not found",
        "duplicate_detail": "invoice already exists",
    },
    {
        "singular": "project",
        "plural": "projects",
        "request_model": "CreateProjectRequest",
        "response_model": "ProjectResponse",
        "list_model": "ProjectSummary",
        "first_field": ("title", "str", "Field(..., min_length=5, max_length=60)"),
        "second_field": ("priority", "int", "Field(..., ge=1, le=5)"),
        "status_field": "state",
        "status_values": ("planned", "active"),
        "query_flag": "include_members",
        "duplicate_value": "duplicate-project",
        "missing_detail": "project not found",
        "duplicate_detail": "project already exists",
    },
    {
        "singular": "ticket",
        "plural": "tickets",
        "request_model": "CreateTicketRequest",
        "response_model": "TicketResponse",
        "list_model": "TicketSummary",
        "first_field": ("subject", "str", "Field(..., min_length=5, max_length=80)"),
        "second_field": ("severity", "int", "Field(..., ge=1, le=4)"),
        "status_field": "status",
        "status_values": ("open", "closed"),
        "query_flag": "include_comments",
        "duplicate_value": "duplicate-ticket",
        "missing_detail": "ticket not found",
        "duplicate_detail": "ticket already exists",
    },
    {
        "singular": "workspace",
        "plural": "workspaces",
        "request_model": "CreateWorkspaceRequest",
        "response_model": "WorkspaceResponse",
        "list_model": "WorkspaceSummary",
        "first_field": ("slug", "str", "Field(..., min_length=3, max_length=24)"),
        "second_field": ("seat_count", "int", "Field(..., ge=1, le=200)"),
        "status_field": "state",
        "status_values": ("trial", "active"),
        "query_flag": "include_audit",
        "duplicate_value": "duplicate-workspace",
        "missing_detail": "workspace not found",
        "duplicate_detail": "workspace already exists",
    },
)

BLUEPRINTS = (
    {
        "template_id": "get_path_response_model",
        "method": "GET",
        "include_path": True,
        "include_query": False,
        "include_body": False,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "single",
        "status_code": 200,
        "error_status": 404,
    },
    {
        "template_id": "get_query_response_model",
        "method": "GET",
        "include_path": False,
        "include_query": True,
        "include_body": False,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "list",
        "status_code": 200,
        "error_status": 400,
    },
    {
        "template_id": "post_body_created_response",
        "method": "POST",
        "include_path": False,
        "include_query": False,
        "include_body": True,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "single",
        "status_code": 201,
        "error_status": 409,
    },
    {
        "template_id": "get_path_and_query_dependency",
        "method": "GET",
        "include_path": True,
        "include_query": True,
        "include_body": False,
        "include_dependency": True,
        "include_auth": True,
        "response_kind": "single",
        "status_code": 200,
        "error_status": 404,
    },
    {
        "template_id": "post_body_query_auth",
        "method": "POST",
        "include_path": False,
        "include_query": True,
        "include_body": True,
        "include_dependency": True,
        "include_auth": True,
        "response_kind": "single",
        "status_code": 201,
        "error_status": 409,
    },
    {
        "template_id": "get_collection_auth",
        "method": "GET",
        "include_path": False,
        "include_query": True,
        "include_body": False,
        "include_dependency": True,
        "include_auth": True,
        "response_kind": "list",
        "status_code": 202,
        "error_status": 400,
    },
    {
        "template_id": "post_nested_body_validation",
        "method": "POST",
        "include_path": False,
        "include_query": False,
        "include_body": True,
        "include_nested_body": True,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "single",
        "status_code": 201,
        "error_status": 409,
    },
    {
        "template_id": "get_status_variant",
        "method": "GET",
        "include_path": True,
        "include_query": False,
        "include_body": False,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "single",
        "status_code": 206,
        "error_status": 404,
    },
    {
        "template_id": "post_body_error_handling",
        "method": "POST",
        "include_path": False,
        "include_query": True,
        "include_body": True,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "single",
        "status_code": 201,
        "error_status": 400,
    },
    {
        "template_id": "get_filtered_collection",
        "method": "GET",
        "include_path": False,
        "include_query": True,
        "include_body": False,
        "include_dependency": True,
        "include_auth": False,
        "response_kind": "list",
        "status_code": 200,
        "error_status": 400,
    },
)


def generate_dataset_bundle(
    *,
    skill_id: str,
    domain: str,
    task_type: str,
    num_examples: int,
    output: str | Path,
    eval_output: str | Path,
    seed: int = DEFAULT_DATASET_SEED,
    eval_size: int | None = None,
    report_output: str | Path | None = None,
) -> dict:
    if num_examples <= 0:
        raise ValueError("--num-examples must be greater than zero")
    resolved_domain = _resolve_domain(domain)
    if task_type != "python_generation":
        raise ValueError("fastapi_contract generation currently supports only python_generation")

    resolved_eval_size = eval_size if eval_size is not None else max(1, round(num_examples * DEFAULT_EVAL_RATIO))
    total_examples = num_examples + resolved_eval_size
    rows = _build_examples(skill_id=skill_id, task_type=task_type, total_examples=total_examples, seed=seed)
    train_rows = _assign_split(rows[:num_examples], split="train", seed=seed, skill_id=skill_id)
    eval_rows = _assign_split(rows[num_examples:], split="eval", seed=seed, skill_id=skill_id)

    output_path = Path(output)
    eval_output_path = Path(eval_output)
    _write_jsonl(output_path, train_rows)
    _write_jsonl(eval_output_path, eval_rows)

    report = validate_training_datasets(output_path, eval_dataset=eval_output_path)
    report["generation"] = {
        "skill_id": skill_id,
        "domain": resolved_domain,
        "task_type": task_type,
        "seed": seed,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "template_distribution": _template_distribution(train_rows + eval_rows),
    }
    report["coverage"] = _coverage_report(train_rows + eval_rows)

    resolved_report = Path(
        report_output or default_report_path(output_path, eval_output=eval_output_path)
    )
    write_validation_report(report, resolved_report)
    if report["errors"]:
        raise RuntimeError(f"generated dataset failed validation; report written to {resolved_report}")

    return {
        "status": report["status"],
        "skill_id": skill_id,
        "domain": resolved_domain,
        "task_type": task_type,
        "seed": seed,
        "train_dataset": str(output_path),
        "eval_dataset": str(eval_output_path),
        "report_output": str(resolved_report),
        "counts": {"train": len(train_rows), "eval": len(eval_rows)},
        "coverage": report["coverage"],
        "warnings": report["warnings"],
    }


def _resolve_domain(domain: str) -> str:
    normalized = domain.strip().lower().replace("-", "_")
    resolved = SUPPORTED_DOMAINS.get(normalized)
    if resolved is None:
        supported = ", ".join(sorted(SUPPORTED_DOMAINS))
        raise ValueError(f"unsupported domain: {domain}; supported values: {supported}")
    return resolved


def _build_examples(*, skill_id: str, task_type: str, total_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    entity_indices = list(range(len(ENTITY_VARIANTS)))
    blueprint_indices = list(range(len(BLUEPRINTS)))
    rng.shuffle(entity_indices)
    rng.shuffle(blueprint_indices)

    rows: list[dict] = []
    for offset in range(total_examples):
        entity_round = offset // len(entity_indices)
        blueprint_round = offset // len(blueprint_indices)
        entity = ENTITY_VARIANTS[
            entity_indices[(offset + entity_round) % len(entity_indices)]
        ]
        blueprint = BLUEPRINTS[
            blueprint_indices[(offset + blueprint_round) % len(blueprint_indices)]
        ]
        variant_number = offset + 1
        rows.append(
            _render_example(
                entity=entity,
                blueprint=blueprint,
                task_type=task_type,
                skill_id=skill_id,
                variant_number=variant_number,
            )
        )
    return rows


def _assign_split(rows: list[dict], *, split: str, seed: int, skill_id: str) -> list[dict]:
    assigned: list[dict] = []
    for index, row in enumerate(rows, 1):
        payload = dict(row)
        payload["id"] = f"{skill_id.replace('_', '-')}-{split}-{index:04d}"
        payload["semantic_family"] = "fastapi_contract"
        payload["skills"] = [skill_id]
        metadata = dict(payload.get("metadata") or {})
        metadata.update({"split": split, "seed": seed, "domain": "fastapi_contract"})
        payload["metadata"] = metadata
        assigned.append(payload)
    return assigned


def _render_example(
    *,
    entity: dict,
    blueprint: dict,
    task_type: str,
    skill_id: str,
    variant_number: int,
) -> dict:
    function_name = _function_name(entity, blueprint, variant_number)
    route_path = _route_path(entity, blueprint)
    prompt = _render_prompt(entity, blueprint, function_name, route_path)
    target = _render_target(entity, blueprint, function_name, route_path, variant_number)
    return {
        "id": f"{skill_id}-{blueprint['template_id']}-{variant_number}",
        "task_type": task_type,
        "prompt": prompt,
        "target": target,
        "metadata": {
            "template_id": blueprint["template_id"],
            "features": _feature_set(blueprint),
        },
    }


def _function_name(entity: dict, blueprint: dict, variant_number: int) -> str:
    if blueprint["method"] == "POST":
        return f"create_{entity['singular']}_{variant_number}"
    if blueprint["response_kind"] == "list":
        return f"list_{entity['plural']}_{variant_number}"
    return f"get_{entity['singular']}_{variant_number}"


def _route_path(entity: dict, blueprint: dict) -> str:
    path = f"/{entity['plural']}"
    if blueprint["include_path"]:
        path += f"/{{{entity['singular']}_id}}"
    if blueprint["method"] == "GET" and blueprint["response_kind"] == "list" and blueprint["include_auth"]:
        path += "/audit"
    if blueprint.get("include_nested_body"):
        path += "/bulk"
    return path


def _render_prompt(entity: dict, blueprint: dict, function_name: str, route_path: str) -> str:
    parts = [
        f"Write FastAPI code for a {blueprint['method']} endpoint named {function_name} mounted at {route_path}.",
        f"Use response_model {_response_model_name(entity, blueprint)} and status code {blueprint['status_code']}.",
    ]
    if blueprint["include_path"]:
        parts.append(f"Validate the path parameter {entity['singular']}_id with Path(..., ge=1).")
    if blueprint["include_query"]:
        parts.append(
            f"Accept query params {entity['query_flag']}: bool and max_{entity['plural']}: int with Query validation."
        )
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


def _render_target(entity: dict, blueprint: dict, function_name: str, route_path: str, variant_number: int) -> str:
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
        lines.extend(
            [
                "",
                "",
                "def require_auth() -> dict[str, str]:",
                '    return {"role": "service"}',
            ]
        )
    if blueprint.get("include_nested_body"):
        lines.extend(
            [
                "",
                "",
                f"class {entity['request_model']}Details(BaseModel):",
                "    region: str = Field(..., min_length=2, max_length=20)",
                "    priority: int = Field(..., ge=1, le=5)",
            ]
        )
    if blueprint["include_body"]:
        first_name, first_type, first_validator = entity["first_field"]
        second_name, second_type, second_validator = entity["second_field"]
        lines.extend(
            [
                "",
                "",
                f"class {entity['request_model']}(BaseModel):",
                f"    {first_name}: {first_type} = {first_validator}",
                f"    {second_name}: {second_type} = {second_validator}",
            ]
        )
        if blueprint.get("include_nested_body"):
            lines.append(f"    details: {entity['request_model']}Details")
    response_model = _response_model_name(entity, blueprint)
    first_name, first_type, first_validator = entity["first_field"]
    status_field = entity["status_field"]
    status_active = entity["status_values"][0]
    status_secondary = entity["status_values"][1]
    lines.extend(
        [
            "",
            "",
            f"class {response_model}(BaseModel):",
            "    id: int = Field(..., ge=1)",
            f"    {first_name}: {first_type} = {first_validator}",
            f"    {status_field}: str",
        ]
    )
    decorator = [
        "",
        "",
        f"@router.{blueprint['method'].lower()}(",
        f'    "{route_path}",',
        f"    response_model={response_model}," if blueprint["response_kind"] == "single" else f"    response_model=list[{response_model}],",
        f"    status_code=status.HTTP_{_status_suffix(blueprint['status_code'])},",
        ")",
    ]
    lines.extend(decorator)
    params = []
    if blueprint["include_path"]:
        params.append(
            f"{entity['singular']}_id: int = Path(..., ge=1, description=\"{entity['singular'].title()} identifier\")"
        )
    if blueprint["include_query"]:
        params.append(
            f"{entity['query_flag']}: bool = Query(False, description=\"Include related objects\")"
        )
        params.append(
            f"max_{entity['plural']}: int = Query(25, ge=1, le=100, description=\"Result size\")"
        )
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
        lines.append(
            f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="access denied")'
        )
    if blueprint["include_path"]:
        lines.append(f"    if {entity['singular']}_id == 9999:")
        lines.append(
            f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="{entity["missing_detail"]}")'
        )
    elif blueprint["include_body"]:
        lines.append(f'    if payload.{entity["first_field"][0]} == "{entity["duplicate_value"]}":')
        lines.append(
            f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="{entity["duplicate_detail"]}")'
        )
    else:
        lines.append(f"    if max_{entity['plural']} < 1:")
        lines.append(
            f'        raise HTTPException(status_code={blueprint["error_status"]}, detail="invalid query")'
        )
    identity_value = f"{entity['singular']}_id" if blueprint["include_path"] else str(variant_number)
    primary_value = f"payload.{entity['first_field'][0]}" if blueprint["include_body"] else f'"{entity["singular"]}-{variant_number}"'
    if blueprint["response_kind"] == "single":
        lines.append(
            f"    return {response_model}(id={identity_value}, {entity['first_field'][0]}={primary_value}, {status_field}=\"{status_active}\")"
        )
    else:
        lines.append("    return [")
        lines.append(
            f"        {response_model}(id={variant_number}, {entity['first_field'][0]}={primary_value}, {status_field}=\"{status_active}\"),"
        )
        lines.append(
            f"        {response_model}(id={variant_number + 1}, {entity['first_field'][0]}=\"{entity['singular']}-{variant_number + 1}\", {status_field}=\"{status_secondary}\"),"
        )
        lines.append("    ]")
    return "\n".join(lines) + "\n"


def _response_model_name(entity: dict, blueprint: dict) -> str:
    return entity["response_model"] if blueprint["response_kind"] == "single" else entity["list_model"]


def _status_suffix(code: int) -> str:
    suffixes = {
        200: "200_OK",
        201: "201_CREATED",
        202: "202_ACCEPTED",
        206: "206_PARTIAL_CONTENT",
    }
    return suffixes[code]


def _feature_set(blueprint: dict) -> list[str]:
    features = ["response_model", "status_codes", "pydantic_validation", "dependency_injection", "error_handling"]
    features.append("get_endpoint" if blueprint["method"] == "GET" else "post_endpoint")
    if blueprint["include_path"]:
        features.append("path_params")
    if blueprint["include_query"]:
        features.append("query_params")
    if blueprint["include_body"]:
        features.append("request_body")
    return sorted(set(features))


def _coverage_report(rows: list[dict]) -> dict:
    counts = {feature: 0 for feature in REQUIRED_FASTAPI_FEATURES}
    for row in rows:
        for feature in row.get("metadata", {}).get("features", []):
            if feature in counts:
                counts[feature] += 1
    missing = [feature for feature, count in counts.items() if count == 0]
    return {
        "required_features": list(REQUIRED_FASTAPI_FEATURES),
        "covered_features": [feature for feature, count in counts.items() if count > 0],
        "missing_features": missing,
        "feature_counts": counts,
    }


def _template_distribution(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        template_id = row.get("metadata", {}).get("template_id", "unknown")
        counts[template_id] = counts.get(template_id, 0) + 1
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))