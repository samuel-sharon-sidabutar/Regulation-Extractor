# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# code-style
- Break up large functions (like extract_main_content and main) into smaller, focused units. Confidence: 0.70
- Prefer guard clauses over long if-else chains for early returns and conditionals. Confidence: 0.80
- Use descriptive action-based function names (e.g., prune_malformed_rows) instead of underscore-prefixed private names (e.g., _handle_malformed_prune). Avoid generic names like run_phase1 that don't convey what the function actually does. Confidence: 0.80

# converter
- Use exponential backoff for API retry cooldowns, starting at 30s initial delay. Confidence: 0.70
- Keep CSV output files directly in the main `regulations/` directory; avoid timestamped subdirectories to make aggregation easier. Confidence: 0.70
- Use a MODEL_PRESETS dict pattern: a single CONFIG["MODEL"] key selects the preset, and each preset holds all model-specific settings (LOG_DIR, MODEL_ID, temperature, etc.), so switching models is a one-word change. Confidence: 0.70

