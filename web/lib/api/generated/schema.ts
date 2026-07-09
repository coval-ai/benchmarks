export interface paths {
    "/healthz": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Healthz
         * @description Liveness probe — always returns 200.
         */
        get: operations["healthz_healthz_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/readyz": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Readyz
         * @description Readiness probe — returns 200 if DB is reachable, 503 otherwise.
         *
         *     Never raises an exception — DB unreachable is the expected failure mode
         *     during Cloud Run startup.
         */
        get: operations["readyz_readyz_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/runs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Runs
         * @description Return a newest-first page of benchmark runs.
         *
         *     Args:
         *         limit: Maximum number of runs to return (1–200, default 50).
         *         before: Cursor — ``id`` of the last run on the previous page.
         *
         *     Returns:
         *         ``{"runs": [...], "next_before": int | None}`` where ``next_before``
         *         is the smallest ``id`` in this page when there are exactly ``limit``
         *         rows, else ``None``.
         */
        get: operations["list_runs_v1_runs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/results": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Results
         * @description Return a newest-first page of successful benchmark results.
         *
         *     All optional filters are ANDed together.
         *
         *     **Result-row vs run-level status distinction:**
         *     ``include_failed`` controls whether results from *failed runs* are included.
         *     The result-row ``status='success'`` filter is always applied — there is no
         *     way to opt in to failed result rows in this phase.
         *
         *     **``metric`` param is deprecated** — use ``metric_type`` for new code.
         *     Both are accepted; if both are provided and equal the request succeeds.
         *     If they differ a 400 is returned.
         *
         *     **``window`` and ``since``/``until`` are mutually exclusive.** Providing both
         *     returns 400.
         */
        get: operations["list_results_v1_results_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/results/aggregates": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Results Aggregates
         * @description Return per-model stats and per-bucket series for one benchmark.
         *
         *     Args:
         *         benchmark: One of STT, TTS.
         *         window: Time window — stats over results.created_at, series over
         *             bucket_at. Defaults to 24h.
         */
        get: operations["get_results_aggregates_v1_results_aggregates_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/leaderboard": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Leaderboard
         * @description Return leaderboard entries sorted ascending by average metric value.
         *
         *     Args:
         *         metric: One of WER, TTFA, TTFT, TTFS.
         *         benchmark: One of STT, TTS.
         *         window: Time window — each is served by its materialized view.
         *
         *     Returns:
         *         ``{"metric": ..., "window": ..., "entries": [LeaderboardEntry, ...]}``
         *
         *     Raises:
         *         400: If the metric/benchmark combination is incompatible.
         */
        get: operations["get_leaderboard_v1_leaderboard_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/providers": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Providers
         * @description Return the catalogue of benchmarked providers and models.
         *
         *     Sourced from the model registry (all entries, not just actively run ones).
         *     Each model includes a ``disabled`` flag that the frontend can use to
         *     hide or grey out models that are known but not actively benchmarked.
         */
        get: operations["get_providers_v1_providers_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /**
         * AggregatesResponse
         * @description Response schema for GET /v1/results/aggregates.
         *
         *     Wraps and returns all our ModelStatEntry and SeriesPoint data for a time
         *     window.
         */
        AggregatesResponse: {
            /**
             * Benchmark
             * @enum {string}
             */
            benchmark: "STT" | "TTS" | "S2S";
            /**
             * Window
             * @enum {string}
             */
            window: "24h" | "7d" | "30d";
            /** Model Stats */
            model_stats: components["schemas"]["ModelStatEntry"][];
            /** Series */
            series: components["schemas"]["SeriesPoint"][];
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /**
         * LeaderboardEntry
         * @description A single entry in the leaderboard response.
         */
        LeaderboardEntry: {
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Avg */
            avg: number;
            /** P50 */
            p50: number;
            /** P95 */
            p95: number;
            /** N */
            n: number;
        };
        /**
         * LeaderboardResponse
         * @description Response schema for GET /v1/leaderboard.
         */
        LeaderboardResponse: {
            /**
             * Metric
             * @enum {string}
             */
            metric: "WER" | "TTFA" | "TTFT" | "TTFS" | "V2V";
            /**
             * Window
             * @enum {string}
             */
            window: "24h" | "7d" | "30d";
            /** Entries */
            entries: components["schemas"]["LeaderboardEntry"][];
        };
        /**
         * ModelInfo
         * @description A single model entry under a provider, with admin-disabled flag.
         */
        ModelInfo: {
            /** Model */
            model: string;
            /**
             * Disabled
             * @default false
             */
            disabled: boolean;
            /**
             * Tags
             * @default []
             */
            tags: components["schemas"]["ModelTagOut"][];
        };
        /**
         * ModelStatEntry
         * @description Per-(provider, model, metric_type) aggregate stats.
         *
         *     Lets us compute the stats server-side and just send the summaries.
         */
        ModelStatEntry: {
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Metric Type */
            metric_type: string;
            /** Avg Value */
            avg_value: number;
            /** Stddev Value */
            stddev_value: number;
            /** P25 */
            p25: number;
            /** P50 */
            p50: number;
            /** P75 */
            p75: number;
            /** P90 */
            p90: number;
            /** P95 */
            p95: number;
            /** P99 */
            p99: number;
            /** Min Value */
            min_value: number;
            /** Max Value */
            max_value: number;
            /** Sample Count */
            sample_count: number;
        };
        /**
         * ModelTagOut
         * @description A faceted filter tag: its category, raw value, and display label.
         */
        ModelTagOut: {
            category: components["schemas"]["TagCategory"];
            /** Value */
            value: string;
            /** Label */
            label: string;
        };
        /**
         * ProviderInfo
         * @description Information about a single provider's models.
         */
        ProviderInfo: {
            /** Provider */
            provider: string;
            /** Models */
            models: components["schemas"]["ModelInfo"][];
            /** Modes */
            modes?: string[] | null;
        };
        /**
         * ProvidersResponse
         * @description Response schema for GET /v1/providers.
         */
        ProvidersResponse: {
            /** Stt */
            stt: components["schemas"]["ProviderInfo"][];
            /** Tts */
            tts: components["schemas"]["ProviderInfo"][];
            /** S2S */
            s2s: components["schemas"]["ProviderInfo"][];
            /** Tag Categories */
            tag_categories: components["schemas"]["TagCategoryOut"][];
        };
        /**
         * ResultOut
         * @description API response schema for a single benchmark result row.
         *
         *     ``status`` is sourced from the *parent run*, not from the result row's own
         *     ``status`` column (which is always ``'success'`` because we filter on it).
         *     The parent-run status is denormalized here at the API boundary via SQL JOIN
         *     so the frontend does not need a second round-trip.
         */
        ResultOut: {
            /** Id */
            id: number;
            /** Run Id */
            run_id: number;
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Voice */
            voice: string | null;
            /**
             * Benchmark
             * @enum {string}
             */
            benchmark: "STT" | "TTS" | "S2S";
            /** Metric Type */
            metric_type: string;
            /** Metric Value */
            metric_value: number | null;
            /** Metric Units */
            metric_units: string | null;
            /** Audio Filename */
            audio_filename: string | null;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /**
             * Scheduled At
             * Format: date-time
             */
            scheduled_at: string;
            /**
             * Status
             * @enum {string}
             */
            status: "RUNNING" | "SUCCEEDED" | "PARTIAL" | "FAILED";
        };
        /**
         * ResultsResponse
         * @description Response schema for GET /v1/results.
         */
        ResultsResponse: {
            /** Results */
            results: components["schemas"]["ResultOut"][];
        };
        /**
         * RunOut
         * @description API response schema for a benchmark run row.
         */
        RunOut: {
            /** Id */
            id: number;
            /**
             * Started At
             * Format: date-time
             */
            started_at: string;
            /** Finished At */
            finished_at: string | null;
            /**
             * Status
             * @enum {string}
             */
            status: "RUNNING" | "SUCCEEDED" | "PARTIAL" | "FAILED";
            /** Runner Sha */
            runner_sha: string;
            /** Dataset Id */
            dataset_id: string;
            /** Dataset Sha256 */
            dataset_sha256: string;
            /** Error */
            error: string | null;
        };
        /**
         * RunsResponse
         * @description Response schema for GET /v1/runs.
         */
        RunsResponse: {
            /** Runs */
            runs: components["schemas"]["RunOut"][];
            /** Next Before */
            next_before?: number | null;
        };
        /**
         * SeriesPoint
         * @description Per-(provider, model, metric_type) distribution for one scheduled_at bucket.
         *
         *     Latency timelines render p50; WER renders value_sum / sample_count.
         */
        SeriesPoint: {
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Metric Type */
            metric_type: string;
            /**
             * Scheduled At
             * Format: date-time
             */
            scheduled_at: string;
            /** Min Value */
            min_value: number;
            /** P25 */
            p25: number;
            /** P50 */
            p50: number;
            /** P75 */
            p75: number;
            /** Max Value */
            max_value: number;
            /** Value Sum */
            value_sum: number;
            /** Sample Count */
            sample_count: number;
        };
        /**
         * TagCategory
         * @description Faceted leaderboard filters. Within a facet tags OR; across facets they AND.
         *
         *     TYPE/HOST/LAB are derived from registry columns and SOURCE/TENANCY/LICENSING/
         *     DEPLOYMENT from model attributes, all at the API boundary; only MODE and
         *     FEATURES draw their values from ``ModelTag``.
         * @enum {string}
         */
        TagCategory: "type" | "mode" | "host" | "lab" | "features" | "source" | "tenancy" | "licensing" | "deployment";
        /**
         * TagCategoryOut
         * @description A facet category's display metadata. Sent in display order.
         */
        TagCategoryOut: {
            category: components["schemas"]["TagCategory"];
            /** Label */
            label: string;
            /**
             * Provider Valued
             * @default false
             */
            provider_valued: boolean;
        };
        /** ValidationError */
        ValidationError: {
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
            /** Input */
            input?: unknown;
            /** Context */
            ctx?: Record<string, never>;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    healthz_healthz_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: string;
                    };
                };
            };
        };
    };
    readyz_readyz_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    list_runs_v1_runs_get: {
        parameters: {
            query?: {
                limit?: number;
                before?: number | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RunsResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_results_v1_results_get: {
        parameters: {
            query?: {
                provider?: string | null;
                model?: string | null;
                /** @description Deprecated alias for metric_type. Kept for backward compatibility with legacy callers. Use metric_type for new integrations. If both are provided and equal, the request succeeds. If they differ, a 400 is returned. */
                metric?: ("WER" | "TTFA" | "TTFT" | "TTFS" | "RTF" | "AUDIO_TO_FINAL" | "V2V") | null;
                /** @description Filter on metric_type (e.g. WER, TTFA, TTFT, RTF). Canonical FE-facing name. */
                metric_type?: string | null;
                benchmark?: ("STT" | "TTS" | "S2S") | null;
                /** @description Time window for results. One of '24h', '7d', '30d'. Defaults to '7d' when neither window nor since/until are supplied. Mutually exclusive with since/until. */
                window?: ("24h" | "7d" | "30d") | null;
                /** @description Lower bound on created_at (ISO 8601). Mutually exclusive with window. */
                since?: string | null;
                /** @description Upper bound on created_at (ISO 8601). May be combined with since. */
                until?: string | null;
                /** @description If false (default), only returns results whose parent run is SUCCEEDED or PARTIAL. If true, results from FAILED and RUNNING parent runs are also included. The result row's own status='success' filter is always applied regardless. */
                include_failed?: boolean;
                /** @description Maximum rows to return (1–100000, default 100000). */
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResultsResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_results_aggregates_v1_results_aggregates_get: {
        parameters: {
            query: {
                benchmark: "STT" | "TTS" | "S2S";
                window?: "24h" | "7d" | "30d";
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AggregatesResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_leaderboard_v1_leaderboard_get: {
        parameters: {
            query: {
                metric: "WER" | "TTFA" | "TTFT" | "TTFS" | "V2V";
                benchmark: "STT" | "TTS" | "S2S";
                window?: "24h" | "7d" | "30d";
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LeaderboardResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_providers_v1_providers_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProvidersResponse"];
                };
            };
        };
    };
}
