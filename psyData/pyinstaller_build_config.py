"""Keep standalone PsySummary bundles limited to runtime dependencies and resources."""


OPTIONAL_BACKEND_EXCLUDES = ('jax', 'jaxlib', 'numba', 'llvmlite', 'pyarrow')
OTHER_QT_BINDINGS = ('PyQt6', 'PySide6', 'PySide2')


def bundle_excludes():
    """Exclude unused optional numerical backends and alternative Qt bindings."""
    return [*OTHER_QT_BINDINGS, *OPTIONAL_BACKEND_EXCLUDES, 'tests', 'app.tests']


def is_bundle_artifact(name, module=False):
    """Identify non-runtime entries without removing scientific testing utilities."""
    parts = name.split('.') if module else [part for part in name.replace('\\', '/').split('/')
                                          if part not in ('', '.')]
    if not parts:
        return False
    if parts[0].split('-')[0] in OPTIONAL_BACKEND_EXCLUDES:
        return True
    directories = parts if module else parts[:-1]
    if directories and directories[0].lower() == 'artifacts':
        return True
    if any(part.lower() in {'tests', '__pycache__', '.pytest_cache'} for part in directories):
        return True
    leaf = parts[-1].lower()
    return not module and (leaf.endswith(('.pyc', '.pyo', '.bak', '~')) or '.bk' in leaf)


def filter_bundle_analysis(analysis):
    """Remove hook-collected test, backup, and optional-backend entries before freezing."""
    for attribute in ('pure', 'datas', 'binaries'):
        entries = getattr(analysis, attribute)
        kept = [entry for entry in entries
                if not is_bundle_artifact(entry[0], module=attribute == 'pure')]
        removed = len(entries) - len(kept)
        if removed:
            print(f'PsySummary packaging: excluded {removed} non-runtime {attribute} entries.')
        setattr(analysis, attribute, kept)
