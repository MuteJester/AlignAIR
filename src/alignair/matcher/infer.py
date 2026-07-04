def called_names(gid: int, groups) -> list:
    """All allele names in the winning unique-sequence group."""
    return list(groups.group_names[gid])
