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
    "/v1/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Health
         * @description Public health check — 200 when the API can serve data, 503 otherwise.
         */
        get: operations["health_v1_health_get"];
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
         * Retired Results
         * @description Return a clear migration response without touching storage or auth state.
         *
         *     Query strings are intentionally not declared: FastAPI ignores arbitrary
         *     legacy filters, so even malformed old values still receive 410.
         */
        get: operations["retired_results_v1_results_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/results": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Results
         * @description Return one successful primary evaluation per result.
         */
        get: operations["list_results_v2_results_get"];
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
         *         benchmark: One of STT, TTS, S2S, LLM.
         *         window: Time window. Defaults to 24h.
         *         dataset: Dataset id the blocks are computed over. Omitted, the pooled
         *             rows (every dataset together) are served — the pre-dataset-dimension
         *             behavior.
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
    "/v1/results/timeline": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Results Timeline
         * @description Return run points or intervals using each metric's aggregation rule.
         */
        get: operations["get_results_timeline_v1_results_timeline_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/results/aggregates/by-dataset": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Results Aggregates By Dataset
         * @description Return per-model stats for every dataset of one benchmark and window.
         *
         *     One block per dataset with data in the window, sorted by dataset id. The
         *     pooled all-dataset rows are not repeated here — the plain aggregates
         *     endpoint serves those.
         */
        get: operations["get_results_aggregates_by_dataset_v1_results_aggregates_by_dataset_get"];
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
         *     Entries under the modality's sample floor are flagged and sink below every
         *     ranked entry, so a model that scored once with a fast time cannot lead.
         *
         *     Args:
         *         metric: One of WER, TTFA, TTFT, TTFS, V2V.
         *         benchmark: One of STT, TTS, S2S, LLM.
         *         window: Time window — each is served by its published snapshot.
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
         *     An early-access model appears only for a caller whose allowlist names it.
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
    "/v1/tags": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Tags
         * @description The FEATURES tag vocabulary, ordered by value.
         */
        get: operations["list_tags_v1_tags_get"];
        put?: never;
        /**
         * Create Tag
         * @description Add a vocabulary entry. There is no delete: models reference tags.
         */
        post: operations["create_tag_v1_tags_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/pricing": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Pricing Registry
         * @description Return every rate in force on ``as_of``, with the earlier spans as history.
         */
        get: operations["get_pricing_registry_v1_pricing_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/s2s/samples": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List S2S Samples
         * @description Sample ids newest-first, empty when the caller may see no S2S model at all.
         */
        get: operations["list_s2s_samples_v1_s2s_samples_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/s2s/samples/{sample_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get S2S Sample
         * @description One sample's manifest, carrying only the recordings this caller may see.
         *
         *     A sample whose every recording is embargoed for this caller is a 404 rather
         *     than an empty shell: the surrounding metadata would otherwise confirm that a
         *     tick exists and name the scenario and persona behind it.
         */
        get: operations["get_s2s_sample_v1_s2s_samples__sample_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/s2s/samples/{sample_id}/{provider}/{model}/audio": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get S2S Sample Audio
         * @description A freshly signed URL for one recording, minted only if this caller may hear it.
         */
        get: operations["get_s2s_sample_audio_v1_s2s_samples__sample_id___provider___model__audio_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/admin/models": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Admin Models
         * @description Every registered model with its state, tags, and recent history.
         */
        get: operations["list_admin_models_v1_admin_models_get"];
        put?: never;
        /**
         * Create Admin Model
         * @description Create a model. It starts collecting immediately; nothing is public yet.
         */
        post: operations["create_admin_model_v1_admin_models_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/admin/models/{model_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /**
         * Update Admin Model
         * @description Partially update a model; the history's ``old`` is what the editor saw.
         */
        patch: operations["update_admin_model_v1_admin_models__model_id__patch"];
        trace?: never;
    };
    "/v1/admin/pricing": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Admin Pricing
         * @description Every model with a pricing log entry: current, scheduled, and the full audit trail.
         */
        get: operations["list_admin_pricing_v1_admin_pricing_get"];
        put?: never;
        /**
         * Record Admin Rate
         * @description Append a recording; 201 when it wrote, 200 when the date already said exactly this.
         */
        post: operations["record_admin_rate_v1_admin_pricing_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/mock/{platform}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Call Tools */
        post: operations["call_tools_mock__platform__post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/mock/{platform}/{tool}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Call Tool */
        post: operations["call_tool_mock__platform___tool__post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/llm/{provider}/session": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Create Session */
        post: operations["create_session_llm__provider__session_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/llm/{provider}/chat": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Chat */
        post: operations["chat_llm__provider__chat_post"];
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
         * AdminChangeOut
         * @description One history row, embedded under its model. ``old`` is NULL on create.
         */
        AdminChangeOut: {
            /** Id */
            id: number;
            /** Old */
            old?: {
                [key: string]: unknown;
            } | null;
            /** New */
            new: {
                [key: string]: unknown;
            };
            /** Changed By User Id */
            changed_by_user_id: string;
            /** Changed By Org Id */
            changed_by_org_id?: string | null;
            /** Changed By Email */
            changed_by_email?: string | null;
            /**
             * Changed At
             * Format: date-time
             */
            changed_at: string;
        };
        /**
         * AdminModelCreate
         * @description POST body for /v1/admin/models. The defaults land the model Hidden.
         */
        AdminModelCreate: {
            modality: components["schemas"]["Benchmark"];
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Voice */
            voice?: string | null;
            /**
             * Voices
             * @default []
             */
            voices: components["schemas"]["Voice"][];
            /** Creator */
            creator?: string | null;
            /** @default official-api */
            source: components["schemas"]["Source"];
            /** @default proprietary */
            licensing: components["schemas"]["Licensing"];
            /**
             * On Prem
             * @default false
             */
            on_prem: boolean;
            /** Region */
            region?: ("us" | "eu" | "asia") | null;
            /**
             * Arena Enabled
             * @default true
             */
            arena_enabled: boolean;
            /**
             * Collected
             * @default true
             */
            collected: boolean;
            /**
             * Published
             * @default false
             */
            published: boolean;
            /**
             * Tags
             * @default []
             */
            tags: string[];
            /** Color */
            color?: string | null;
            /** Display Name */
            display_name?: string | null;
            architecture?: components["schemas"]["Architecture"] | null;
            orchestration?: components["schemas"]["Orchestration"] | null;
        };
        /**
         * AdminModelOut
         * @description A registered model with full state, provenance, and recent history.
         */
        AdminModelOut: {
            /** Id */
            id: number;
            modality: components["schemas"]["Benchmark"];
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Voice */
            voice?: string | null;
            /**
             * Voices
             * @default []
             */
            voices: components["schemas"]["Voice"][];
            /** Creator */
            creator?: string | null;
            source: components["schemas"]["Source"];
            licensing: components["schemas"]["Licensing"];
            /** On Prem */
            on_prem: boolean;
            /** Region */
            region?: ("us" | "eu" | "asia") | null;
            /** Arena Enabled */
            arena_enabled: boolean;
            /** Collected */
            collected: boolean;
            /** Published */
            published: boolean;
            /**
             * Tags
             * @default []
             */
            tags: string[];
            /** Color */
            color?: string | null;
            /** Display Name */
            display_name?: string | null;
            architecture?: components["schemas"]["Architecture"] | null;
            orchestration?: components["schemas"]["Orchestration"] | null;
            /** Updated By User Id */
            updated_by_user_id: string;
            /** Updated By Email */
            updated_by_email?: string | null;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
            /**
             * History
             * @default []
             */
            history: components["schemas"]["AdminChangeOut"][];
        };
        /**
         * AdminModelPatch
         * @description PATCH body for /v1/admin/models/{id}; absent fields stay unchanged.
         *
         *     ``color`` null returns the model to the site's built-in palette; ``display_name``
         *     null returns it to the site's built-in label.
         */
        AdminModelPatch: {
            /** Provider */
            provider?: string | null;
            /** Model */
            model?: string | null;
            /** Voice */
            voice?: string | null;
            /** Voices */
            voices?: components["schemas"]["Voice"][] | null;
            /** Creator */
            creator?: string | null;
            source?: components["schemas"]["Source"] | null;
            licensing?: components["schemas"]["Licensing"] | null;
            /** On Prem */
            on_prem?: boolean | null;
            /** Region */
            region?: ("us" | "eu" | "asia") | null;
            /** Arena Enabled */
            arena_enabled?: boolean | null;
            /** Collected */
            collected?: boolean | null;
            /** Published */
            published?: boolean | null;
            /** Tags */
            tags?: string[] | null;
            /** Color */
            color?: string | null;
            /** Display Name */
            display_name?: string | null;
            architecture?: components["schemas"]["Architecture"] | null;
            orchestration?: components["schemas"]["Orchestration"] | null;
        };
        /**
         * AdminModelPricingOut
         * @description One model's pricing log: in force today, scheduled, and every recording newest first.
         */
        AdminModelPricingOut: {
            benchmark: components["schemas"]["Benchmark"];
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            current?: components["schemas"]["AdminRateRecordingOut"] | null;
            /**
             * Scheduled
             * @default []
             */
            scheduled: components["schemas"]["AdminRateRecordingOut"][];
            /**
             * Recordings
             * @default []
             */
            recordings: components["schemas"]["AdminRateRecordingOut"][];
        };
        /**
         * AdminModelUpdateResponse
         * @description PATCH response: the updated model plus warnings a rename leaves behind.
         */
        AdminModelUpdateResponse: {
            model: components["schemas"]["AdminModelOut"];
            /**
             * Warnings
             * @default []
             */
            warnings: string[];
        };
        /**
         * AdminModelsResponse
         * @description Response schema for GET /v1/admin/models.
         */
        AdminModelsResponse: {
            /** Models */
            models: components["schemas"]["AdminModelOut"][];
        };
        /**
         * AdminPricingResponse
         * @description Response schema for GET /v1/admin/pricing.
         */
        AdminPricingResponse: {
            /**
             * As Of
             * Format: date
             */
            as_of: string;
            /** Models */
            models: components["schemas"]["AdminModelPricingOut"][];
        };
        /**
         * AdminRateCreate
         * @description POST body for /v1/admin/pricing: ``unit`` + ``price_usd`` (decimal string) for a
         *     published rate, neither for a delisting; ``effective_from`` defaults to today (UTC).
         */
        AdminRateCreate: {
            benchmark: components["schemas"]["Benchmark"];
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Unit */
            unit?: string | null;
            /** Price Usd */
            price_usd?: string | null;
            /** Effective From */
            effective_from?: string | null;
            /** Source Url */
            source_url?: string | null;
            /** Notes */
            notes?: string | null;
        };
        /**
         * AdminRateRecordingOut
         * @description One row of the pricing log as the admin sees it; ``superseded`` marks a corrected entry.
         */
        AdminRateRecordingOut: {
            /** Unit */
            unit?: string | null;
            /** Price Usd */
            price_usd?: string | null;
            /** Price Per 1M Chars */
            price_per_1m_chars?: number | null;
            /** Price Per 1K Minutes */
            price_per_1k_minutes?: number | null;
            /**
             * Effective From
             * Format: date
             */
            effective_from: string;
            /** Effective To */
            effective_to?: string | null;
            /** Source Url */
            source_url?: string | null;
            /** Id */
            id: number;
            /** Notes */
            notes?: string | null;
            /** Recorded By User Id */
            recorded_by_user_id: string;
            /** Recorded By Email */
            recorded_by_email?: string | null;
            /**
             * Recorded At
             * Format: date-time
             */
            recorded_at: string;
            /** Superseded */
            superseded: boolean;
        };
        /**
         * AggregatesByDatasetResponse
         * @description Response schema for GET /v1/results/aggregates/by-dataset.
         *
         *     One block per dataset with data in the window, sorted by dataset id.
         *     Series are deliberately absent: per-dataset timelines would multiply the
         *     payload by the dataset count and nothing consumes them batched — the
         *     single-dataset endpoint serves that long tail. ``primary_dataset`` is the
         *     dataset the board's headline reads; None for boards without one.
         */
        AggregatesByDatasetResponse: {
            /**
             * Benchmark
             * @enum {string}
             */
            benchmark: "STT" | "TTS" | "S2S" | "LLM";
            /**
             * Window
             * @enum {string}
             */
            window: "24h" | "7d" | "30d";
            /** Primary Dataset */
            primary_dataset?: string | null;
            /** Blocks */
            blocks: components["schemas"]["DatasetAggregates"][];
            snapshot?: components["schemas"]["DashboardSnapshot"] | null;
        };
        /**
         * AggregatesResponse
         * @description Response schema for GET /v1/results/aggregates.
         *
         *     Wraps and returns all our ModelStatEntry and SeriesPoint data for a time
         *     window. ``dataset`` echoes the dataset the blocks were computed over
         *     ('__all__' = pooled across datasets); ``datasets`` lists the dataset ids
         *     with data in the window. ``primary_dataset`` is the dataset the board's
         *     headline reads; None for boards without one.
         */
        AggregatesResponse: {
            /**
             * Benchmark
             * @enum {string}
             */
            benchmark: "STT" | "TTS" | "S2S" | "LLM";
            /**
             * Window
             * @enum {string}
             */
            window: "24h" | "7d" | "30d";
            /** Dataset */
            dataset: string;
            /** Datasets */
            datasets: string[];
            /** Primary Dataset */
            primary_dataset?: string | null;
            /** Model Stats */
            model_stats: components["schemas"]["ModelStatEntry"][];
            /** Series */
            series: components["schemas"]["SeriesPoint"][];
            snapshot?: components["schemas"]["DashboardSnapshot"] | null;
        };
        /**
         * Architecture
         * @description How an S2S agent makes speech: one model, or speech-to-text, an LLM and text-to-speech.
         * @enum {string}
         */
        Architecture: "speech-to-speech" | "cascaded";
        /**
         * Benchmark
         * @description Which benchmark a model or result belongs to.
         * @enum {string}
         */
        Benchmark: "STT" | "TTS" | "S2S" | "LLM";
        /** ChatRequest */
        ChatRequest: {
            /** Model */
            model: string;
            /** Messages */
            messages: {
                [key: string]: unknown;
            }[];
            /** Simulation Id */
            simulation_id?: string | null;
            /** Stream */
            stream?: boolean | null;
        };
        /** DashboardMaterialization */
        DashboardMaterialization: {
            /** Refreshed At */
            refreshed_at?: string | null;
            /**
             * Stale
             * @default false
             */
            stale: boolean;
        };
        /** DashboardSnapshot */
        DashboardSnapshot: {
            /** Generation */
            generation: number;
            /**
             * As Of
             * Format: date-time
             */
            as_of: string;
            /**
             * Published At
             * Format: date-time
             */
            published_at: string;
            /** Definition Revision */
            definition_revision: number;
            /** Stale */
            stale: boolean;
        };
        /**
         * DatasetAggregates
         * @description Per-model stats for one dataset — one block of the by-dataset response.
         */
        DatasetAggregates: {
            /** Dataset */
            dataset: string;
            /** Model Stats */
            model_stats: components["schemas"]["ModelStatEntry"][];
            persona?: components["schemas"]["DatasetPersona"] | null;
        };
        /**
         * DatasetPersona
         * @description The caller persona a dataset was run with, as declared in the runner's registry.
         */
        DatasetPersona: {
            /** Slug */
            slug: string;
            /** Label */
            label: string;
            /** Description */
            description: string;
            /** Order */
            order: number;
            /**
             * Anchor
             * @enum {string}
             */
            anchor: "latency" | "judge";
        };
        /**
         * Gender
         * @description Perceived gender of a voice — an editorial label, not ground truth.
         *
         *     Provider metadata is too uneven to encode provenance: Hume tags every stock
         *     voice, OpenAI publishes none. Four providers here are API-confirmed
         *     (cartesia, hume, xai, lmnt); the rest are our own listening.
         * @enum {string}
         */
        Gender: "female" | "male";
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
            /**
             * Insufficient Samples
             * @default false
             */
            insufficient_samples: boolean;
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
            snapshot?: components["schemas"]["DashboardSnapshot"] | null;
        };
        /**
         * Licensing
         * @description Whether the model's weights are openly available.
         * @enum {string}
         */
        Licensing: "proprietary" | "open-weight";
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
             * Early Access
             * @default false
             */
            early_access: boolean;
            /**
             * Tags
             * @default []
             */
            tags: components["schemas"]["ModelTagOut"][];
            /** Color */
            color?: string | null;
            /** Display Name */
            display_name?: string | null;
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
            /** Mean Value */
            mean_value?: number | null;
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
            /** Min Value */
            min_value: number;
            /** Max Value */
            max_value: number;
            /** Sample Count */
            sample_count: number;
            /**
             * Insufficient Samples
             * @default false
             */
            insufficient_samples: boolean;
            /** Wer Insertions Pct */
            wer_insertions_pct?: number | null;
            /** Wer Deletions Pct */
            wer_deletions_pct?: number | null;
            /** Wer Substitutions Pct */
            wer_substitutions_pct?: number | null;
            /** Pooled Value */
            pooled_value?: number | null;
            /** Pooled Insertions Pct */
            pooled_insertions_pct?: number | null;
            /** Pooled Deletions Pct */
            pooled_deletions_pct?: number | null;
            /** Pooled Substitutions Pct */
            pooled_substitutions_pct?: number | null;
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
         * Orchestration
         * @description Who runs an S2S agent's turn-taking and tools: our harness, or a hosted platform.
         * @enum {string}
         */
        Orchestration: "harness" | "platform";
        /**
         * PricingRateOut
         * @description One model's rate in force on the requested day, with the earlier spans oldest first.
         */
        PricingRateOut: {
            /** Unit */
            unit?: string | null;
            /** Price Usd */
            price_usd?: string | null;
            /** Price Per 1M Chars */
            price_per_1m_chars?: number | null;
            /** Price Per 1K Minutes */
            price_per_1k_minutes?: number | null;
            /**
             * Effective From
             * Format: date
             */
            effective_from: string;
            /** Effective To */
            effective_to?: string | null;
            /** Source Url */
            source_url?: string | null;
            benchmark: components["schemas"]["Benchmark"];
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Notes */
            notes?: string | null;
            /**
             * Recorded At
             * Format: date-time
             */
            recorded_at: string;
            /**
             * History
             * @default []
             */
            history: components["schemas"]["PricingRateSpanOut"][];
        };
        /**
         * PricingRateSpanOut
         * @description One stretch of a model's price history.
         *
         *     ``price_usd`` is the provider's native figure in ``unit`` as the exact decimal
         *     string it was recorded as; the normalized fields are None where no conversion
         *     exists without assuming a speaking rate, and all three are None over a span
         *     with no known public rate. ``effective_to`` is exclusive, None while open.
         */
        PricingRateSpanOut: {
            /** Unit */
            unit?: string | null;
            /** Price Usd */
            price_usd?: string | null;
            /** Price Per 1M Chars */
            price_per_1m_chars?: number | null;
            /** Price Per 1K Minutes */
            price_per_1k_minutes?: number | null;
            /**
             * Effective From
             * Format: date
             */
            effective_from: string;
            /** Effective To */
            effective_to?: string | null;
            /** Source Url */
            source_url?: string | null;
        };
        /**
         * PricingRegistryResponse
         * @description Response schema for GET /v1/pricing: the rates in force on ``as_of``.
         */
        PricingRegistryResponse: {
            /**
             * As Of
             * Format: date
             */
            as_of: string;
            /** Rates */
            rates: components["schemas"]["PricingRateOut"][];
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
            /** Llm */
            llm: components["schemas"]["ProviderInfo"][];
            /** Tag Categories */
            tag_categories: components["schemas"]["TagCategoryOut"][];
        };
        /** ResultComponentOut */
        ResultComponentOut: {
            /** Value */
            value: number;
            /** Unit */
            unit: string;
        };
        /** ResultV2Out */
        ResultV2Out: {
            /** Metric Type */
            metric_type: string;
            /** Metric Version */
            metric_version: string;
            /** Value */
            value: number;
            /** Unit */
            unit: string;
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
            benchmark: "STT" | "TTS" | "S2S" | "LLM";
            /** Dataset Id */
            dataset_id: string;
            /**
             * Captured At
             * Format: date-time
             */
            captured_at: string;
            /** Components */
            components?: {
                [key: string]: components["schemas"]["ResultComponentOut"];
            };
        };
        /** ResultsV2Response */
        ResultsV2Response: {
            /** Results */
            results: components["schemas"]["ResultV2Out"][];
            /** Next Cursor */
            next_cursor: string | null;
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
         * S2SSampleAudioOut
         * @description A freshly signed URL for one recording, with the moment it stops working.
         *
         *     Handed over as a body rather than a redirect: a browser cannot carry its
         *     early-access header through a cross-origin redirect to storage, so the caller
         *     fetches this with its proof and then points an audio element at ``url``.
         *     ``expires_at`` lets the caller re-ask before playing rather than after failing.
         */
        S2SSampleAudioOut: {
            /** Url */
            url: string;
            /**
             * Expires At
             * Format: date-time
             */
            expires_at: string;
        };
        /**
         * S2SSampleOut
         * @description One sample, filtered to the recordings this caller may see.
         */
        S2SSampleOut: {
            /** Schema Version */
            schema_version?: number | null;
            /** Sample Id */
            sample_id: string;
            /** Dataset Id */
            dataset_id?: string | null;
            /** Test Case Id */
            test_case_id: string;
            /** Test Set Id */
            test_set_id?: string | null;
            /** Persona Name */
            persona_name?: string | null;
            /** Transcript */
            transcript?: string | null;
            /** Recordings */
            recordings: components["schemas"]["S2SSampleRecordingOut"][];
        };
        /**
         * S2SSampleRecordingOut
         * @description One model's recording. ``audio_path`` is an API route, not a storage URL.
         */
        S2SSampleRecordingOut: {
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Audio Path */
            audio_path: string;
            /** Coval Run Id */
            coval_run_id: string;
            /** Sim Id */
            sim_id: string;
            /** Agent Id */
            agent_id?: string | null;
            /** Turns */
            turns?: components["schemas"]["S2SSampleTurnOut"][];
        };
        /**
         * S2SSampleTurnOut
         * @description One spoken turn of a sampled conversation.
         */
        S2SSampleTurnOut: {
            /** Index */
            index: number;
            /** Role */
            role: string;
            /** Content */
            content: string;
            /** Start Offset */
            start_offset?: number | null;
            /** End Offset */
            end_offset?: number | null;
        };
        /**
         * SeriesPoint
         * @description Per-(provider, model, metric_type) distribution for one scheduled_at bucket.
         *
         *     Latency timelines render p50. WER renders error_sum / wer_reference_words when
         *     present, else value_sum / sample_count; both pairs sum across buckets.
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
            /** Error Sum */
            error_sum?: number | null;
            /** Reference Word Sum */
            reference_word_sum?: number | null;
            /** Pooled Value */
            pooled_value?: number | null;
        };
        /**
         * Source
         * @description Where the benchmarked endpoint lives: the creator's own API or an inference host.
         * @enum {string}
         */
        Source: "official-api" | "shared-inference" | "dedicated-inference";
        /**
         * TagCategory
         * @description Faceted leaderboard filters. Within a facet tags OR; across facets they AND.
         *
         *     TYPE/HOST/CREATOR are derived from registry columns and SOURCE/LICENSING/
         *     DEPLOYMENT from model attributes, all at the API boundary; FEATURES values
         *     and labels come from ``benchmarks_v2.tags``. ARCHITECTURE and ORCHESTRATION
         *     are S2S-only columns, absent from a model that leaves them unset.
         * @enum {string}
         */
        TagCategory: "type" | "host" | "creator" | "features" | "source" | "licensing" | "deployment" | "region" | "architecture" | "orchestration";
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
        /**
         * TagOut
         * @description A FEATURES vocabulary entry.
         */
        TagOut: {
            /** Value */
            value: string;
            /**
             * Category
             * @constant
             */
            category: "features";
            /** Label */
            label: string;
        };
        /**
         * TagsResponse
         * @description Response schema for GET /v1/tags.
         */
        TagsResponse: {
            /** Tags */
            tags: components["schemas"]["TagOut"][];
        };
        /**
         * TimelinePoint
         * @description One compact chart point for a provider/model/metric bucket.
         */
        TimelinePoint: {
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
            /** Value */
            value: number | null;
            /** Pooled Value */
            pooled_value?: number | null;
            /** Aggregation Method */
            aggregation_method?: ("mean" | "ratio" | "mean_fallback" | "unavailable" | "percentile") | null;
            /** Sample Count */
            sample_count?: number | null;
            /**
             * Insufficient Samples
             * @default false
             */
            insufficient_samples: boolean;
        };
        /**
         * TimelineResponse
         * @description Response schema for GET /v1/results/timeline.
         */
        TimelineResponse: {
            /**
             * Benchmark
             * @enum {string}
             */
            benchmark: "STT" | "TTS" | "S2S" | "LLM";
            /**
             * Statistic
             * @default default
             * @enum {string}
             */
            statistic: "default" | "p50" | "p90" | "p95" | "p100";
            /** Metric Type */
            metric_type?: string | null;
            /** Precision */
            precision?: "exact" | null;
            /** Percentile Method */
            percentile_method?: "continuous" | null;
            /** Weighting */
            weighting?: "observation" | null;
            /** Window */
            window: ("24h" | "7d" | "30d") | null;
            /** Dataset */
            dataset: string;
            /** Points */
            points: components["schemas"]["TimelinePoint"][];
            /**
             * Aggregation
             * @default run
             * @enum {string}
             */
            aggregation: "run" | "average";
            /** Bucket Seconds */
            bucket_seconds?: number | null;
            /** Range Start */
            range_start?: string | null;
            /** Range End */
            range_end?: string | null;
            /** Latest Source At */
            latest_source_at?: string | null;
            snapshot?: components["schemas"]["DashboardSnapshot"] | null;
            materialization?: components["schemas"]["DashboardMaterialization"] | null;
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
        /**
         * Voice
         * @description One speaker, as the provider addresses it.
         *
         *     ``id`` goes on the wire and is often an opaque UUID, so ``name`` keeps
         *     registry diffs readable. ``accent`` is recorded where known, not controlled.
         *
         *     Nothing pairs on ``gender`` yet: the arena synthesizes with the scalar
         *     ``RegisteredModel.voice``, which has no gender and for most models isn't
         *     even a pool member. Battles today are cross-gender by construction.
         */
        Voice: {
            /** Id */
            id: string;
            gender: components["schemas"]["Gender"];
            /** Name */
            name?: string | null;
            /** Accent */
            accent?: string | null;
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
    health_v1_health_get: {
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
    retired_results_v1_results_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The legacy row-level results reader has been retired. */
            410: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    /**
                     * @example {
                     *       "detail": "The legacy /v1/results endpoint has been retired.",
                     *       "replacement": "/v2/results"
                     *     }
                     */
                    "application/json": unknown;
                };
            };
        };
    };
    list_results_v2_results_get: {
        parameters: {
            query?: {
                run_id?: number | null;
                provider?: string | null;
                model?: string | null;
                benchmark?: ("STT" | "TTS" | "S2S" | "LLM") | null;
                dataset?: string | null;
                metric_type?: string | null;
                /** @description Exact metric version; omitted returns all versions. */
                metric_version?: string | null;
                /** @description Only default-variant evaluations are public. */
                evaluation_variant?: "default";
                /** @description Only successful evaluations are public. */
                evaluation_status?: "succeeded";
                /** @description Parent run state; omitted includes succeeded and partial. */
                run_status?: ("succeeded" | "partial") | null;
                include_components?: boolean;
                limit?: number;
                /** @description Relative capture window; defaults to 7d only without run_id or bounds. */
                window?: ("24h" | "7d" | "30d") | null;
                /** @description Inclusive timezone-aware capture bound; excludes window. */
                since?: string | null;
                /** @description Exclusive timezone-aware capture bound; excludes window. */
                until?: string | null;
                /** @description Opaque token from the prior page. */
                cursor?: string | null;
            };
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": components["schemas"]["ResultsV2Response"];
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
                benchmark: "STT" | "TTS" | "S2S" | "LLM";
                window?: "24h" | "7d" | "30d";
                /** @description Dataset id to aggregate over; omit for the pooled all-dataset blocks. */
                dataset?: string | null;
                /** @description Whether to include the per-bucket chart series. */
                include_series?: boolean;
            };
            header?: {
                authorization?: string | null;
            };
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
    get_results_timeline_v1_results_timeline_get: {
        parameters: {
            query: {
                benchmark: "STT" | "TTS" | "S2S" | "LLM";
                window?: ("24h" | "7d" | "30d") | null;
                statistic?: "default" | "p50" | "p90" | "p95" | "p100";
                metric_type?: string | null;
                /** @description Dataset id to aggregate over; omit for pooled all-dataset buckets. */
                dataset?: string | null;
                since?: string | null;
                until?: string | null;
            };
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": components["schemas"]["TimelineResponse"];
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
    get_results_aggregates_by_dataset_v1_results_aggregates_by_dataset_get: {
        parameters: {
            query: {
                benchmark: "STT" | "TTS" | "S2S" | "LLM";
                window?: "24h" | "7d" | "30d";
            };
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": components["schemas"]["AggregatesByDatasetResponse"];
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
                benchmark: "STT" | "TTS" | "S2S" | "LLM";
                window?: "24h" | "7d" | "30d";
            };
            header?: {
                authorization?: string | null;
            };
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
            header?: {
                authorization?: string | null;
            };
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
    list_tags_v1_tags_get: {
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
                    "application/json": components["schemas"]["TagsResponse"];
                };
            };
        };
    };
    create_tag_v1_tags_post: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["TagOut"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["TagOut"];
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
    get_pricing_registry_v1_pricing_get: {
        parameters: {
            query?: {
                /** @description The day to price as of (UTC); defaults to today and may not be later. */
                as_of?: string | null;
            };
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": components["schemas"]["PricingRegistryResponse"];
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
    list_s2s_samples_v1_s2s_samples_get: {
        parameters: {
            query?: {
                /** @description Samples partition (an S2S dataset id); omitted reads the legacy root layout. */
                dataset?: string | null;
            };
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": string[];
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
    get_s2s_sample_v1_s2s_samples__sample_id__get: {
        parameters: {
            query?: {
                /** @description Samples partition (an S2S dataset id); omitted reads the legacy root layout. */
                dataset?: string | null;
            };
            header?: {
                authorization?: string | null;
            };
            path: {
                sample_id: string;
            };
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
                    "application/json": components["schemas"]["S2SSampleOut"];
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
    get_s2s_sample_audio_v1_s2s_samples__sample_id___provider___model__audio_get: {
        parameters: {
            query?: {
                /** @description Samples partition (an S2S dataset id); omitted reads the legacy root layout. */
                dataset?: string | null;
            };
            header?: {
                authorization?: string | null;
            };
            path: {
                provider: string;
                model: string;
                sample_id: string;
            };
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
                    "application/json": components["schemas"]["S2SSampleAudioOut"];
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
    list_admin_models_v1_admin_models_get: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": components["schemas"]["AdminModelsResponse"];
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
    create_admin_model_v1_admin_models_post: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AdminModelCreate"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AdminModelOut"];
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
    update_admin_model_v1_admin_models__model_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "if-match"?: string | null;
                authorization?: string | null;
            };
            path: {
                model_id: number;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AdminModelPatch"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AdminModelUpdateResponse"];
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
    list_admin_pricing_v1_admin_pricing_get: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
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
                    "application/json": components["schemas"]["AdminPricingResponse"];
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
    record_admin_rate_v1_admin_pricing_post: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AdminRateCreate"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AdminModelPricingOut"];
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
    call_tools_mock__platform__post: {
        parameters: {
            query?: {
                suite?: string | null;
            };
            header?: {
                "x-mock-tools-key"?: string | null;
            };
            path: {
                platform: string;
            };
            cookie?: never;
        };
        requestBody?: {
            content: {
                "application/json": {
                    [key: string]: unknown;
                } | null;
            };
        };
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
    call_tool_mock__platform___tool__post: {
        parameters: {
            query?: {
                suite?: string | null;
            };
            header?: {
                "x-mock-tools-key"?: string | null;
            };
            path: {
                platform: string;
                tool: string;
            };
            cookie?: never;
        };
        requestBody?: {
            content: {
                "application/json": {
                    [key: string]: unknown;
                } | null;
            };
        };
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
    create_session_llm__provider__session_post: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
            path: {
                provider: string;
            };
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
                        [key: string]: string | null;
                    };
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
    chat_llm__provider__chat_post: {
        parameters: {
            query?: never;
            header?: {
                authorization?: string | null;
            };
            path: {
                provider: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ChatRequest"];
            };
        };
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
}
