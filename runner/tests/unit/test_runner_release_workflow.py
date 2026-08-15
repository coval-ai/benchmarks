import pathlib
import textwrap

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github" / "workflows" / "runner-release.yml"


def _step_script(workflow: str, step_name: str) -> str:
    step_marker = f"      - name: {step_name}\n"
    step_start = workflow.find(step_marker)
    if step_start < 0:
        raise AssertionError(f"Workflow step not found: {step_name}")

    run_marker = "        run: |\n"
    script_start = workflow.find(run_marker, step_start)
    if script_start < 0:
        raise AssertionError(f"Workflow script not found: {step_name}")

    content_start = script_start + len(run_marker)
    next_step = workflow.find("\n      - name:", content_start)
    script = workflow[content_start:] if next_step < 0 else workflow[content_start:next_step]
    return textwrap.dedent(script)


def test_switchboard_dispatch_accepts_only_the_mapped_production_environment() -> None:
    script = _step_script(_WORKFLOW_PATH.read_text(), "Validate release environment")

    assert "repository_dispatch)" in script
    assert 'if [ "$DISPATCH_ENVIRONMENT" != "production" ]; then' in script
    assert "Benchmarks runner release environment must be production" in script


def test_manual_production_release_remains_supported() -> None:
    script = _step_script(_WORKFLOW_PATH.read_text(), "Validate release environment")

    assert "workflow_dispatch) ;;" in script


def test_unknown_release_event_fails_closed() -> None:
    script = _step_script(_WORKFLOW_PATH.read_text(), "Validate release environment")

    assert "*)" in script
    assert "Unsupported Benchmarks runner release event" in script


def test_environment_validation_precedes_checkout() -> None:
    workflow = _WORKFLOW_PATH.read_text()

    assert workflow.index("- name: Validate release environment") < workflow.index(
        "- name: Checkout repository"
    )
