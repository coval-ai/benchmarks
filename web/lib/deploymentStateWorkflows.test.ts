import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const webRelease = fs.readFileSync(
  path.join(repositoryRoot, ".github/workflows/web-release.yml"),
  "utf8",
);
const runnerRelease = fs.readFileSync(
  path.join(repositoryRoot, ".github/workflows/runner-release.yml"),
  "utf8",
);
const promotionFoundationSha =
  "a973276e4639129a06cef4cf92409d4aa8e1e305";

describe("Benchmarks deployment-state workflows", () => {
  it("preserves immutable release context for both services", () => {
    expect(webRelease).toContain("benchmarks.web.production.release");
    expect(webRelease).toContain(
      "RESOLVED_SHA: ${{ steps.revision.outputs.resolved_sha }}",
    );
    expect(runnerRelease).toContain("benchmarks.runner.production.release");
    expect(runnerRelease).toContain(
      "SOURCE_SHA: ${{ needs.context.outputs.deploy_sha }}",
    );
  });

  it("records successful web and runner releases through the shared contract", () => {
    expect(webRelease).toContain('service: "benchmarks.web"');
    expect(webRelease).toContain('kind: "vercel_deployment"');
    expect(runnerRelease).toContain('service: "benchmarks.runner"');
    expect(runnerRelease).toContain(
      'id: "benchmarks.runner.infrastructure-rollout"',
    );
    expect(webRelease).toContain("name: switchboard-deploy-result");
    expect(runnerRelease).toContain("name: switchboard-deploy-result");
  });

  it("seeds state only after a successful manual deployment", () => {
    expect(webRelease).toContain(
      "needs.context.outputs.promotion_origin == 'manual'",
    );
    expect(webRelease).toContain("steps.release.outputs.deployed == 'true'");
    expect(runnerRelease).toContain(
      "needs.context.outputs.promotion_origin == 'manual'",
    );
    expect(runnerRelease).toContain("needs.context.outputs.skip != 'true'");
    expect(webRelease).toContain(`ref: ${promotionFoundationSha}`);
    expect(runnerRelease).toContain(`ref: ${promotionFoundationSha}`);
  });

  it("does not activate automatic-promotion enforcement", () => {
    expect(webRelease).not.toContain("switchboard-automatic-promotion");
    expect(runnerRelease).not.toContain("switchboard-automatic-promotion");
  });
});
