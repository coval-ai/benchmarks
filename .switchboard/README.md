# Runner and API input ownership

`impact.json` publishes runner/CI and API-type-generation input subscriptions as
data. Runtime changes under `runner/` affect both surfaces. The generator script,
artifact manifest and publication workflow affect API types; runner workflows,
methodology inputs and the new declaration files retain runner validation.

Repository documentation and the manual arena launcher are classified separately
from these subscriptions. Inventory classification is not proof of runtime consumer
absence. Every other new path must fit a declared input or fail required source
validation; there is no catchall passive input masking future code or workflows.

`service-selection-contracts.json` asserts existing runtime, methodology, workflow,
generator, manifest, declaration, documentation and mixed-file cases through the
shared evaluator. It contains no commands, check/gate policy or callbacks. Reader
activation must qualify the actual landed declaration and its complete source
inventory. Native execution, benchmark scheduling and publication behavior remain
with their existing workflows; no paid run is introduced by this metadata.

## Native test catalog

`catalog-generation.json` declares the physical test files under `runner/` and
asserts the `coval_bench` import root from the runner's Hatch project. The
catalog publishes the broad runner source scope and leaves individual tests
unscoped. The full-suite cases cover runner source, tests, project and pytest
configuration, CI and catalog metadata.

The catalog declaration itself is a runner CI input. API type generation keeps
its independent inputs. Native pytest collection, plugin and fixture behavior,
execution and release gates remain with their current workflows. Switchboard
can require this catalog after qualifying its landed commit.
