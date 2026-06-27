from .constants import REQUIRED_FASTAPI_FEATURES


def response_model_name(entity: dict, blueprint: dict) -> str:
    return entity["response_model"] if blueprint["response_kind"] == "single" else entity["list_model"]


def status_suffix(code: int) -> str:
    suffixes = {
        200: "200_OK",
        201: "201_CREATED",
        202: "202_ACCEPTED",
        206: "206_PARTIAL_CONTENT",
    }
    return suffixes[code]


def feature_set(blueprint: dict) -> list[str]:
    features = ["response_model", "status_codes", "pydantic_validation", "dependency_injection", "error_handling"]
    features.append("get_endpoint" if blueprint["method"] == "GET" else "post_endpoint")
    if blueprint["include_path"]:
        features.append("path_params")
    if blueprint["include_query"]:
        features.append("query_params")
    if blueprint["include_body"]:
        features.append("request_body")
    return sorted(set(features))


def coverage_report(rows: list[dict]) -> dict:
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


def template_distribution(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        template_id = row.get("metadata", {}).get("template_id", "unknown")
        counts[template_id] = counts.get(template_id, 0) + 1
    return dict(sorted(counts.items()))
