# Timeline aggregation

The timeline separates the metric's aggregation rule from the displayed time
interval. Changing the interval must not change the meaning of the metric.

## Metric rules

Each versioned metric value contract declares one of these rules:

- **Mean:** add the primary value sums and divide by the total observation count.
- **Ratio:** add each declared numerator component, divide by the total declared
  denominator component, then apply the scale. Component totals must cover all
  primary observations in every source bucket.

For example, phonetic accuracy defined as correctly matched phonemes divided by
reference phonemes would declare `correct / reference * 100`. Clips with 9/10 and
50/100 correct phonemes combine to 59/110, or about 53.6%. Averaging the two clip
percentages would give 70% and answer a different question. This example does
not register a production phonetic metric.

Missing components, incomplete coverage, and a zero denominator do not produce
a ratio. The chart leaves a gap unless the metric explicitly permits a fallback.
WER preserves its historical mean-of-clip-percentages fallback, identified as
`mean_fallback`. When counts are complete, WER pools substitutions, deletions,
and insertions over reference words. A displayed interval uses one method for
all its observations; it never averages a mixture of pooled ratios and means.

The API includes the method and observation count for averaged points. Counts
refer to primary observations, not the ratio's denominator (for example, clips
versus phonemes). Failed evaluations are excluded under existing eligibility
rules; this count is not a success rate or the number of attempted requests.

The rule is versioned with the emitted values. Display units and ranking
direction remain in the metric display registry. Adding a metric still requires
its normal computation and display registration, but mean/ratio timeline
aggregation does not require another metric-specific SQL branch.

## Time ranges and zoom

- The 24-hour preset retains its per-run values and local zoom behavior.
- Seven days uses one-hour intervals; thirty days uses four-hour intervals.
- Averaged views choose a supported interval for roughly 200 chart points, with
  one hour as the minimum interval.
- Drag selections on averaged views snap to whole UTC hours within the available
  range, with a minimum one-hour selection. Further crops reuse loaded data when
  it already has the required resolution and covers the selection.
- The API continues to accept exact explicit bounds. Source timestamps use
  half-open bounds, `since <= source_at < until`; the response echoes those
  bounds. Tooltips identify the portion of each interval covered by the request.

Data remains separated by provider, model, metric, and the requested dataset.
Current dashboard reads select metric version `v1` and evaluation variant
`default`; adding another version must not silently combine the two.

These rules apply to averaged timelines. Existing headline summaries and the
legacy aggregate-series endpoint retain their existing contracts.

## Percentile follow-up

A percentile selector is separate work. Existing summary queries already
calculate exact percentiles from individual values. Timeline buckets retain
fixed quartiles, extrema, sums, and counts; those values cannot reconstruct p90
or p95 across a larger interval. Averaging stored percentiles is not valid.

The mean/ratio change requires no database migration. A follow-up can calculate
percentiles from retained individual observations without inherently changing
the schema. If query measurements justify storing a mergeable distribution
summary, that approach needs its own migration, writer changes, and a decision
about backfilling historical data. It does not block mean/ratio aggregation.

That follow-up should define the observation unit and weighting, exact versus
approximate results, low-sample labels, and failure handling. Latency and WER
benefit from upper percentiles such as p90/p95; higher-is-better accuracy can use
lower percentiles such as p5/p10 to expose poorly performing clips.
