# Fixtures of the mock hub

The responses of the Hub API that the quantization menu reads, captured so the menu can be tested
offline.

- `models/` holds the info of a repo: its files and their size, its `base_model` metadata, the
  parameters of its weights…
- `configs/` holds the `config.json` of a repo, which holds the dimensions of the model.
- `manifest.json` holds the order the Hub listed the repos in, so that the menu of the mock hub is
  the menu of the Hub, down to the order of the rows of the same size.
- `extra/` holds hand written fixtures, for the cases the Hub does not have a model for (an
  adapter, a repo with several serializations of its weights…).

## Refreshing them

```bash
pnpm --filter @huggingface/hub dump-fixtures -- org/model
```

A model is captured with the variants published for it, which are discovered from the Hub, the way
the menu discovers them: they are the repos that declare the model in their `base_model` metadata.
Several models can be captured at once:

```bash
pnpm --filter @huggingface/hub dump-fixtures -- org/model other/model
```

The number of variants captured for a model is the number the menu reads by default. It can be
raised:

```bash
pnpm --filter @huggingface/hub dump-fixtures -- org/model --limit 100
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
pnpm --filter @huggingface/hub quant-variants -- org/model --hub-url http://localhost:8787
```

Or let the menu start the mock hub for the time of the script:

```bash
pnpm --filter @huggingface/hub quant-variants -- org/model --offline
```

## Seeing them on a model page

The mock site serves the pages of the Hub, built from the API of the mock hub:

```bash
pnpm --filter @huggingface/hub mock-site
```

- `http://localhost:8787/` lists the models of the fixtures.
- `http://localhost:8787/<owner>/<model>` shows a model page, with the quantization menu of the
  model.

The hardware the menu is compared to is chosen with the `hardware` query parameter, and the
memory of a custom hardware with the `memory` one:

```bash
http://localhost:8787/org/model?hardware=NVIDIA%20RTX%204090
http://localhost:8787/org/model?memory=64
```
