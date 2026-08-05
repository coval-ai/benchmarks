// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { BarDataPoint, TtfaBreakdownBar } from "../../../types/benchmark.types";
import CustomBarTooltip, { TtfaBreakdownTooltip } from "./BarTooltip";

const point: BarDataPoint = {
  model: "soniox/stt-rt-v5",
  averageWER: 6.6,
  provider: "Soniox",
  breakdown: { substitutions: 4.1, deletions: 1.5, insertions: 1.0 },
};

const render = (p: BarDataPoint) =>
  renderToStaticMarkup(
    <CustomBarTooltip
      active
      payload={[{ dataKey: "averageWER", value: p.averageWER, payload: p }] as never}
      label={p.model}
    />
  );

describe("CustomBarTooltip", () => {
  it("lists each error type under the total", () => {
    const html = render(point);
    expect(html).toContain("Average WER: 6.6%");
    expect(html).toContain("Substitutions: 4.1%");
    expect(html).toContain("Deletions: 1.5%");
    expect(html).toContain("Insertions: 1.0%");
  });

  it("falls back to the total alone when the API sends no split", () => {
    const html = render({ ...point, breakdown: undefined });
    expect(html).toContain("Average WER: 6.6%");
    expect(html).not.toContain("Substitutions");
    expect(html).not.toContain("Deletions");
    expect(html).not.toContain("Insertions");
  });
});

describe("TtfaBreakdownTooltip", () => {
  it("leads with the total and gives each segment its share", () => {
    const bar: TtfaBreakdownBar = {
      model: "elevenlabs:eleven_flash_v2_5",
      provider: "elevenlabs",
      roundtrip: 160,
      silence: 40,
      ttfa: 200,
    };
    const html = renderToStaticMarkup(
      <TtfaBreakdownTooltip
        active
        payload={[{ dataKey: "roundtrip", value: bar.roundtrip, payload: bar }] as never}
        label={bar.model}
      />
    );
    expect(html).toContain("Avg TTFA: 200 ms");
    expect(html).toContain("Network roundtrip: 160 ms (80%)");
    expect(html).toContain("Leading silence: 40 ms (20%)");
  });

  it("omits shares when the total is zero instead of rendering NaN%", () => {
    const bar: TtfaBreakdownBar = {
      model: "elevenlabs:eleven_flash_v2_5",
      provider: "elevenlabs",
      roundtrip: 0,
      silence: 0,
      ttfa: 0,
    };
    const html = renderToStaticMarkup(
      <TtfaBreakdownTooltip
        active
        payload={[{ dataKey: "roundtrip", value: 0, payload: bar }] as never}
        label={bar.model}
      />
    );
    expect(html).toContain("Network roundtrip: 0 ms");
    expect(html).not.toContain("NaN");
    expect(html).not.toContain("%");
  });
});
