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
