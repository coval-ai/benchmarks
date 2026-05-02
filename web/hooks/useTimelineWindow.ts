// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

"use client";

import { useState, useCallback, useEffect, useRef } from "react";

interface TimelineDataPoint {
  timestamp: number;
  [key: string]: string | number | boolean | null | undefined;
}

interface UseTimelineWindowProps {
  initialEnd: number;
  getTimelineData: () => TimelineDataPoint[];
  visibleWindowMs: number;
}

export const useTimelineWindow = ({
  initialEnd,
  getTimelineData,
  visibleWindowMs
}: UseTimelineWindowProps) => {
  const [timelineWindowEnd, setTimelineWindowEnd] = useState<number>(initialEnd);
  // Capture the chart element at mousedown time so drag math uses the chart
  // the user actually clicked. Prior implementation shared a single chartRef
  // between TimelineChart and PerformanceDeltaSection — last-mounted wins —
  // which made drag distance scale wrong when the wrong rect was used.
  const dragElementRef = useRef<HTMLElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartTime, setDragStartTime] = useState(0);

  const getFullTimeRange = useCallback((): [number, number] => {
    const allTimelineData = getTimelineData().map((item) => item.timestamp);

    if (allTimelineData.length === 0)
      return [Date.now() - visibleWindowMs, Date.now()];

    return [Math.min(...allTimelineData), Math.max(...allTimelineData)];
  }, [getTimelineData, visibleWindowMs]);

  const handleMouseDown = (e: React.MouseEvent<HTMLElement>) => {
    dragElementRef.current = e.currentTarget;
    setIsDragging(true);
    setDragStartX(e.clientX);
    setDragStartTime(timelineWindowEnd);
    e.preventDefault();
  };

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      const el = dragElementRef.current;
      if (!isDragging || !el) return;

      const rect = el.getBoundingClientRect();
      const deltaX = e.clientX - dragStartX;
      const chartWidth = rect.width;
      if (chartWidth === 0) return;
      const timeRange = visibleWindowMs;

      const timeDelta = (deltaX / chartWidth) * timeRange;
      const newEnd = dragStartTime - timeDelta;

      const [fullMin, fullMax] = getFullTimeRange();
      const clampedEnd = Math.max(
        fullMin + visibleWindowMs,
        Math.min(newEnd, fullMax)
      );

      setTimelineWindowEnd(clampedEnd);
    },
    [isDragging, dragStartX, dragStartTime, visibleWindowMs, getFullTimeRange]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragElementRef.current = null;
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return {
    timelineWindowEnd,
    setTimelineWindowEnd,
    isDragging,
    dragStartX,
    dragStartTime,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp
  };
};
