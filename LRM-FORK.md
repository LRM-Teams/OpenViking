# LRM fork of OpenViking

This repository is LRM-Teams' fork of [volcengine/OpenViking](https://github.com/volcengine/OpenViking)
(AGPL-3.0). It carries the causal-memory work line and builds the image CoForge's
**non-production** OpenViking prototype runs.

- Upstream is `volcengine/OpenViking`. The fork's `main` merges upstream `main`
  and keeps the causal-memory commits on top; it does not rewrite upstream
  history, so upstream revisions CoForge pins stay reachable here.
- `feat/fork-node-foundation` is the active causal-memory branch (evaluation
  loop, applicability verification, delivery throttle, outcome utility).
- `openviking/_version.py` is checked in from upstream. The image build passes
  `OPENVIKING_VERSION` explicitly, so the version never depends on local tags.

## License and deployment stance

Upstream is AGPL-3.0 and CoForge's project-level license review for a shipping
integration is still **pending**. Everything built here is non-production:
synthetic data only, no staging and no production deployment, no Caddy route.
Do not present the prototype image as a shippable CoForge capability.

## CI

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `lrm-ci.yml` | push to `main`, pull requests to `main`, manual | Mirrors upstream `_test_lite.yml` (uv sync, `cargo check -p ragfs-python`, in-place native build, cuVS CPU regression tests) and then runs the fork's causal-memory test files listed in the workflow. |

`lrm-ci.yml` duplicates the steps of upstream `_test_lite.yml` on purpose: this
repository cannot run upstream's `pr.yml` (see below), and a `workflow_call` to
`_test_lite.yml` would build the environment twice. **When upstream changes
`_test_lite.yml`, copy the change into `lrm-ci.yml`.** The same workflow lists
every fork test file explicitly — add new fork test files there.

## CD

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `lrm-prototype-image.yml` | push to `main` touching source or build inputs, `openviking-prototype-v*` tags, manual | Builds the unmodified upstream `Dockerfile` for `linux/amd64`, pushes `ghcr.io/lrm-teams/openviking`, then boots the pushed digest with a synthetic config and waits for `/health` on port 1933. Prints the immutable pin in the job summary and uploads it as the `prototype-image-digest` artifact. |

CoForge pins the digest, never a moving tag:

```bash
export COFORGE_OPENVIKING_PROTOTYPE_IMAGE=ghcr.io/lrm-teams/openviking@sha256:<digest>
```

GHCR packages start **private** even in a public repository, and a personal
access token without `read:packages` cannot change that. Pulling the digest on
a machine therefore needs one of:

- `echo "$GHCR_PAT" | docker login ghcr.io -u <user> --password-stdin`, then
  pull with a token that has `read:packages`; or
- an org owner sets the package to public in the organization's package
  settings (Package settings → Change visibility) so an anonymous
  `docker pull` works. The CI smoke job authenticates with `GITHUB_TOKEN`, so it
  does not depend on this choice.

`infra/docker/openviking-prototype-local-embed.Dockerfile` in CoForge adds the
`llama-cpp-python` layer; point its `FROM` at a digest from this repository to
make the documented prototype image reproducible instead of hand-built.

## Upstream workflows that are disabled here

Upstream ships ~28 workflows. The ones that need upstream secrets, paid or
unavailable runners, or release targets that are not this fork's business are
**disabled in the repository settings** rather than deleted, so the tree stays
identical to upstream and merges stay conflict-free.

Disabled in the repository settings (not deleted):

- `01. Pull Request Checks` — its `_build.yml` matrix needs macOS, Windows, and
  `ubuntu-24.04-arm` runners and builds distributions this fork does not ship.
- `03. Release`, `15. _Build Distribution`, `16. _Publish Distribution`, `20.
  Release TOS Upload` — they publish PyPI/npm/TOS artifacts for upstream.
- `04. Weekly Security Scan` — `02` already runs the same CodeQL scan on `main`.
- `05. OpenClaw2OpenViking Memory Tests`, `06. API & CLI Integration Tests`,
  `07. API Effect Tests` — they run on schedules and on release and need the
  `EMBEDDING_API_KEY` / `VLM_API_KEY` secrets.
- `13. _Test Suite (Full)` — upstream's full multi-OS suite.
- `17. Docs`, `18. _Docs Deploy`, `19. Docs TOS Deploy` — they deploy docs to
  upstream's Pages and TOS bucket (`TOS_*` secrets).
- `Build and Push Docker Image` — pushes multi-arch to GHCR **and Docker Hub**
  (`DOCKERHUB_*` secrets). `lrm-prototype-image.yml` is this fork's replacement.
- `Rust CLI Build`, `Plugin npm Release`, `Plugin Shared Runtime Sync`,
  `TypeScript SDK Release`, `Python SDK Release`, `LangChain Integration
  Release`, `Controlplane MCP Release`, `OpenViking OpenClaw plugin release`,
  `Star Stats` — upstream release/publishing plumbing (`NPM_TOKEN`,
  `CLAWHUB_TOKEN`) and the star-history job.

Enabled on purpose: `02. Main Branch Checks` with `14. _CodeQL Scan`
(upstream's own CodeQL gate, free on a public repository), `12. _Test Suite
(Lite)` (kept dispatchable, and the reference `lrm-ci.yml` mirrors), and the two
`lrm-*` workflows above.

List and re-enable from the CLI:

```bash
gh workflow list --repo LRM-Teams/OpenViking --all
gh workflow enable "<workflow name>" --repo LRM-Teams/OpenViking
```

Re-enabling is safe for any workflow that needs no secrets. The ones that
reference `DOCKERHUB_*`, `TOS_*`, `NPM_TOKEN`, `CLAWHUB_TOKEN`,
`EMBEDDING_API_KEY`, or `VLM_API_KEY` fail without those secrets configured on
this repository.

## Merging upstream

```bash
git remote add upstream https://github.com/volcengine/OpenViking.git   # once
git fetch upstream
git merge upstream/main     # from the fork's main
```

No fork file shadows an upstream path: all fork additions (`lrm-*` workflows,
`LRM-FORK.md`) are new paths, so upstream merges do not conflict on them. If
upstream ever adds a file with the same name, rename the fork's instead of
editing upstream's.
