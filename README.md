# interactor-taskweft-function-block-diagram-teacher

The compiler-labelled training corpus for a teacher model that answers only in IEC 61131-3 Function Block Diagrams.

## What it is for

The writers in `tools/` construct diagram rows, and the compiler and sandbox runners in `lib/` label them. The repository holds no model, training or inference; RFD66606-2236 owns the teacher's design.

## Building and running

```sh
pixi run write-rows
mix test
```

The pixi environment covers Windows and Linux x86_64 only. `write-rows` builds the training corpus; the pixi tasks name the other corpus families and the publish steps.

## Licence

MIT. See [LICENSE](LICENSE).
