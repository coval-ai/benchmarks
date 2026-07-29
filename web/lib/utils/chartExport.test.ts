// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { serializeCSV } from "./csv";

describe("serializeCSV", () => {
  it("keeps later-row columns and leaves missing values blank", () => {
    expect(serializeCSV([{ model: "a" }, { model: "b", runs: 2 }])).toBe(
      "model,runs\r\na,\r\nb,2"
    );
  });

  it("escapes quotes, commas, and line breaks", () => {
    expect(
      serializeCSV([{ model: 'A, "quoted"', detail: "line 1\r\nline 2" }])
    ).toBe('model,detail\r\n"A, ""quoted""","line 1\r\nline 2"');
  });

  it("exports null and non-finite values as blank cells", () => {
    expect(
      serializeCSV([{ a: null, b: undefined, c: NaN, d: Infinity }])
    ).toBe("a,b,c,d\r\n,,,");
  });
});
