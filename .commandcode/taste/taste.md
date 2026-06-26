# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# converter
- Use exponential backoff for API retry cooldowns, starting at 30s initial delay. Confidence: 0.70
- Keep CSV output files directly in the main `regulations/` directory; avoid timestamped subdirectories to make aggregation easier. Confidence: 0.70
- Use a MODEL_PRESETS dict pattern: a single CONFIG["MODEL"] key selects the preset, and each preset holds all model-specific settings (LOG_DIR, MODEL_ID, temperature, etc.), so switching models is a one-word change. Confidence: 0.70

