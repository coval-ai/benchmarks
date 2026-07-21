// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { forwardRef, useImperativeHandle } from "react";
import {
  useActiveTooltipCoordinate,
  useActiveTooltipLabel,
  usePlotArea,
  useXAxisInverseScale,
  useYAxisInverseScale,
  type ActiveLabel,
  type Coordinate,
  type PlotArea,
} from "recharts";

export interface ChartInteractionHandle {
  plotArea?: PlotArea;
  xValueAt: (pixel: number) => unknown;
  yValueAt: (pixel: number) => unknown;
  tooltip: {
    label: ActiveLabel;
    coordinate?: Coordinate;
  };
}

const ChartInteractionLayer = forwardRef<ChartInteractionHandle>(
  function ChartInteractionLayer(_, ref) {
    const plotArea = usePlotArea();
    const xScale = useXAxisInverseScale();
    const yScale = useYAxisInverseScale();
    const label = useActiveTooltipLabel();
    const coordinate = useActiveTooltipCoordinate();

    useImperativeHandle(
      ref,
      () => ({
        plotArea,
        xValueAt: (pixel) => xScale?.(pixel),
        yValueAt: (pixel) => yScale?.(pixel),
        tooltip: { label, coordinate },
      }),
      [plotArea, xScale, yScale, label, coordinate]
    );

    return null;
  }
);

export default ChartInteractionLayer;
