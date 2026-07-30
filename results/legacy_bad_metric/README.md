# Invalidated: Bad Metrics

These results were invalidated because `total_seen` represented the
resident KV before final eviction, not the original logical context
capacity. The end-to-end ratio = kept / logical_tokens, not kept / total_seen.

Migrated to `_retention_utils.summarize_retention()` which uses
`logical_tokens_by_req` as denominator.

Date invalidated: 2026-07-30
