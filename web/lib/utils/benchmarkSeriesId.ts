/**
 * Composite benchmark series identity: `provider` + `model` from the API/DB.
 * Used for dashboard selection and Recharts keys so two vendors sharing the
 * same matrix slug (e.g. `default`) never merge.
 */

export const BENCHMARK_SERIES_SEP = "\u001f";

export function encodeBenchmarkSeriesId(provider: string, model: string): string {
  if (provider.includes(BENCHMARK_SERIES_SEP) || model.includes(BENCHMARK_SERIES_SEP)) {
    throw new Error(
      "encodeBenchmarkSeriesId: provider and model must not contain U+001F"
    );
  }
  return `${provider}${BENCHMARK_SERIES_SEP}${model}`;
}

export function decodeBenchmarkSeriesId(id: string): { provider: string; model: string } {
  const i = id.indexOf(BENCHMARK_SERIES_SEP);
  if (
    i <= 0 ||
    i === id.length - 1 ||
    id.indexOf(BENCHMARK_SERIES_SEP, i + 1) !== -1
  ) {
    throw new Error(`decodeBenchmarkSeriesId: invalid id ${JSON.stringify(id.slice(0, 80))}`);
  }
  return { provider: id.slice(0, i), model: id.slice(i + 1) };
}

export function tryDecodeBenchmarkSeriesId(
  id: string
): { provider: string; model: string } | null {
  const i = id.indexOf(BENCHMARK_SERIES_SEP);
  if (
    i <= 0 ||
    i === id.length - 1 ||
    id.indexOf(BENCHMARK_SERIES_SEP, i + 1) !== -1
  ) {
    return null;
  }
  return { provider: id.slice(0, i), model: id.slice(i + 1) };
}

export function rowMatchesSeriesId(
  row: { provider: string; model: string },
  seriesId: string
): boolean {
  return encodeBenchmarkSeriesId(row.provider, row.model) === seriesId;
}

export const SERIES_VALUE_DATA_SUFFIX = "__v";
export const SERIES_GAP_DATA_SUFFIX = "__g";

export function seriesValueDataKey(seriesId: string): string {
  return `${seriesId}${SERIES_VALUE_DATA_SUFFIX}`;
}

export function seriesGapDataKey(seriesId: string): string {
  return `${seriesId}${SERIES_GAP_DATA_SUFFIX}`;
}

export function parseSeriesValueDataKey(dataKey: string): string | null {
  if (!dataKey.endsWith(SERIES_VALUE_DATA_SUFFIX)) return null;
  return dataKey.slice(0, -SERIES_VALUE_DATA_SUFFIX.length);
}

export function parseSeriesGapDataKey(dataKey: string): string | null {
  if (!dataKey.endsWith(SERIES_GAP_DATA_SUFFIX)) return null;
  return dataKey.slice(0, -SERIES_GAP_DATA_SUFFIX.length);
}
