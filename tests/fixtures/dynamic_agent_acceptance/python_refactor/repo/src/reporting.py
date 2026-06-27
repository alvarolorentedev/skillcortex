def render_report(rows):
    lines = []
    for row in rows:
        lines.append(f"{row['name']}: {row['value']}")
    return "\n".join(lines)


def render_summary(rows):
    lines = []
    for row in rows:
        lines.append(f"{row['name']} -> {row['value']}")
    return "\n".join(lines)
