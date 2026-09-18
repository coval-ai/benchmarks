import pathlib

REPOSITORY_ROOT = pathlib.Path(__file__).parents[3]


def test_switchboard_release_dispatches_built_image_digests() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "runner-release.yml").read_text()

    assert "IMAGE_DIGEST: ${{ steps.build.outputs.digest }}" in workflow
    assert "needs: [context, resolve_images]" in workflow
    assert "RUNNER_IMAGE: ${{ needs.resolve_images.outputs.runner_image }}" in workflow
    assert "API_IMAGE: ${{ needs.resolve_images.outputs.api_image }}" in workflow
    assert "coval-bench-runner@${runner_digest}" in workflow
    assert "coval-bench-api@${api_digest}" in workflow


def test_switchboard_release_watches_only_its_correlated_infra_run() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "runner-release.yml").read_text()

    assert 'endswith(" " + $correlation)' in workflow
    assert "for _ in $(seq 1 30)" in workflow
    assert 'gh run view "$candidate_id"' not in workflow
    assert "newest repository_dispatch run" not in workflow
    assert "fallback_run_id" not in workflow
    assert "for correlation ${correlation_id}" in workflow


def test_tag_release_preserves_tags_but_dispatches_built_digests() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "tag.yml").read_text()

    assert "coval-bench-${{ matrix.image }}:${{ github.ref_name }}" in workflow
    assert "IMAGE_DIGEST: ${{ steps.build.outputs.digest }}" in workflow
    assert '"runner_image": "${{ needs.resolve-images.outputs.runner_image }}"' in workflow
    assert '"api_image": "${{ needs.resolve-images.outputs.api_image }}"' in workflow
    assert "coval-bench-runner@${runner_digest}" in workflow
    assert "coval-bench-api@${api_digest}" in workflow
