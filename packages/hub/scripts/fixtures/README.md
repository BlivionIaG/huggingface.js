# Fixtures of the mock hub

The responses of the Hub API that the quantization menu reads, captured so the menu can be tested
offline.

- `models/` holds the info of a repo: its files and their size, its `base_model` metadata, the
  parameters of its weights…
- `configs/` holds the `config.json` of a repo, which holds the dimensions of the model.
- `extra/` holds hand written fixtures, for the cases the Hub does not have a model for (an
  adapter, a repo with several serializations of its weights…).

## Refreshing them

```bash
pnpm --filter @huggingface/hub dump-fixtures
```

The models captured by default are listed in `scripts/dump-fixtures.ts`. Other models can be
captured by name:

```bash
pnpm --filter @huggingface/hub exec tsx scripts/dump-fixtures.ts org/model
```

## Editing them

The fixtures are plain JSON, and can be edited to test a case: change the size of a file to see
the menu react, add a variant to see it listed, remove a `base_model` to see it ignored.

## Serving them

```bash
pnpm --filter @huggingface/hub mock-hub
```

Then point the menu at the mock hub:

```bash
pnpm --filter @huggingface/hub quant-variants -- Qwen/Qwen3.8-27B --hub-url http://localhost:8787
```
