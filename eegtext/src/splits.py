"""Leakage checks shared by future retrieval and sentiment splits."""


def assert_disjoint_text_groups(partitions):
    """Reject a text group appearing in more than one named partition."""

    owners = {}
    overlaps = {}
    for name, group_ids in partitions.items():
        for group_id in set(group_ids):
            previous = owners.setdefault(group_id, name)
            if previous != name:
                overlaps.setdefault(group_id, {previous}).add(name)
    if overlaps:
        examples = ", ".join(
            f"{group_id[:12]}:{'/'.join(sorted(names))}"
            for group_id, names in sorted(overlaps.items())[:5]
        )
        raise ValueError(f"text groups cross partitions: {examples}")
    return True
