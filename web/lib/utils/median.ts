// Copyright 2026 The Coval Benchmarks Authors
// SPDX-License-Identifier: Apache-2.0

function selectKth(arr: number[], k: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const a = arr[lo]!;
    const b = arr[mid]!;
    const c = arr[hi]!;
    const pivot =
      a < b ? (b < c ? b : a < c ? c : a) : a < c ? a : b < c ? c : b;
    let i = lo - 1;
    let j = hi + 1;
    for (;;) {
      do i++;
      while (arr[i]! < pivot);
      do j--;
      while (arr[j]! > pivot);
      if (i >= j) break;
      const tmp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = tmp;
    }
    if (k <= j) hi = j;
    else lo = j + 1;
  }
  return arr[k]!;
}

export function median(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  const mid = n >> 1;
  if (n % 2 === 1) return selectKth(values.slice(), mid);
  return (selectKth(values.slice(), mid - 1) + selectKth(values.slice(), mid)) / 2;
}
