"""Project directory scan: report which ready pipeline artifacts exist on disk."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectArtifacts:
    """Paths of ready project artifacts; ``None`` where the file is absent."""

    nifti: Path | None = None
    fiducials: Path | None = None
    measurements: Path | None = None
    coordsystem: Path | None = None
    pipeline_config: Path | None = None
    head_mask: Path | None = None
    base_mesh: Path | None = None
    scalp_mesh: Path | None = None
    scalp_points: Path | None = None
    ese_mesh: Path | None = None
    ese_points: Path | None = None


@dataclass(frozen=True)
class ArtifactRule:
    """One scan rule: a file pattern mapped to a :class:`ProjectArtifacts` slot.

    ``folders`` is the tuple of project subdirectories to search, or
    ``None`` to search the whole project recursively. ``exclude`` lists
    file names the rule never matches, so rules with overlapping patterns
    (e.g. the NIfTI glob and the head mask) stay unambiguous.
    """

    slot: str
    """``ProjectArtifacts`` field the first matching file fills."""
    pattern: str
    """File name (or glob) matched against candidate paths."""
    folders: tuple[str, ...] | None = None
    """Subdirectories to search; ``None`` searches the whole project."""
    exclude: tuple[str, ...] = ()
    """File names this rule never matches."""


#: Single source of truth for artifact discovery. Rules run in order and
#: the first match fills its slot; every variant name is a first-class
#: rule (no canonical-folder preference, no fallbacks). Most names are
#: searched anywhere in the project; ``mesh.ply`` counts as an ESE mesh
#: only under ``ese/``, and ``*.nii.gz`` outranks ``*.nii`` (the head
#: mask is excluded from the NIfTI globs). When several
#: files match one rule, the first in sorted path order wins.
#: Dot-directories are never searched.
_ARTIFACT_RULES: tuple[ArtifactRule, ...] = (
    ArtifactRule("nifti", "*.nii.gz", exclude=("head_mask.nii.gz",)),
    ArtifactRule("nifti", "*.nii"),
    ArtifactRule("fiducials", "fiducials.json"),
    ArtifactRule("measurements", "measurements.json"),
    ArtifactRule("coordsystem", "coordsystem.json"),
    ArtifactRule("pipeline_config", "pipeline_config.json"),
    ArtifactRule("head_mask", "head_mask.nii.gz"),
    ArtifactRule("base_mesh", "base_mesh.ply"),
    ArtifactRule("scalp_mesh", "final_mesh.ply"),
    ArtifactRule("ese_mesh", "mesh.ply", ("ese",)),
    ArtifactRule("ese_mesh", "ese_mesh.ply"),
    ArtifactRule("scalp_points", "electrodes_scalp.csv"),
    ArtifactRule("ese_points", "electrodes_ese.csv"),
)


def _candidates(project: Path, rule: ArtifactRule) -> Iterator[Path]:
    """Yield the rule's candidate files in deterministic order.

    Whole-project rules walk the tree recursively; named-folder rules
    match the pattern one level inside each folder. Files under a
    dot-directory are skipped.
    """
    if rule.folders is None:
        found = sorted(project.rglob(rule.pattern))
    else:
        found = [
            path
            for folder in rule.folders
            for path in sorted((project / folder).glob(rule.pattern))
        ]
    for path in found:
        if path.name in rule.exclude:
            continue
        if not any(part.startswith(".") for part in path.relative_to(project).parts[:-1]):
            yield path


def find_project_artifacts(project: Path) -> ProjectArtifacts:
    """Scan a VIRDA project directory for ready artifact files.

    Scanning only: file contents are not read. The rules in
    :data:`_ARTIFACT_RULES` run in order and the first match fills its
    slot; see that constant for the searched file names and folders.
    Dot-directories (``.venv``, ``.opencode``, ...) are skipped.
    """
    found: dict[str, Path] = {}
    for rule in _ARTIFACT_RULES:
        if rule.slot in found:
            continue
        path = next(_candidates(project, rule), None)
        if path is not None:
            found[rule.slot] = path
    return ProjectArtifacts(**found)
