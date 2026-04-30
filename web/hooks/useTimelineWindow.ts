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
  const chartRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartTime, setDragStartTime] = useState(0);

  const getFullTimeRange = useCallback((): [number, number] => {
    const allTimelineData = getTimelineData().map((item) => item.timestamp);

    if (allTimelineData.length === 0)
      return [Date.now() - visibleWindowMs, Date.now()];

    return [Math.min(...allTimelineData), Math.max(...allTimelineData)];
  }, [getTimelineData, visibleWindowMs]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStartX(e.clientX);
    setDragStartTime(timelineWindowEnd);
    e.preventDefault();
  };

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !chartRef.current) return;

      const rect = chartRef.current.getBoundingClientRect();
      const deltaX = e.clientX - dragStartX;
      const chartWidth = rect.width;
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
    chartRef,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp
  };
};
