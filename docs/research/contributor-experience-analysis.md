# Contributor Experience Analysis: Open-Source Inference/AI Projects

> **Research Date**: July 2026  
> **Purpose**: Understand how top open-source inference projects structure their contributor experience to inform Synapse's contributor strategy.

---

## Table of Contents

1. [vLLM](#1-vllm)
2. [SGLang](#2-sglang)
3. [Exo](#3-exo)
4. [Petals](#4-petals)
5. [llama.cpp](#5-llamacpp)
6. [Ollama](#6-ollama)
7. [Best Practices Summary & Recommendations for Synapse](#7-best-practices-summary--recommendations-for-synapse)

---

## 1. vLLM

**Repository**: [vllm-project/vllm](https://github.com/vllm-project/vllm)  
**Stars**: 87.8k | **Forks**: 20.1k | **Commits**: 19,377 | **Contributors**: 2,000+

### GitHub Description
> "A high-throughput and memory-efficient inference and serving engine for LLMs"

### Topics
`amd`, `blackwell`, `cuda`, `deepseek`, `deepseek-v3`, `gpt`, `gpt-oss`, `inference`, `kimi`, `llama`, `llm`, `llm-serving`, `model-serving`, `moe`, `openai`, `pytorch`, `qwen`, `qwen3`, `tpu`, `transformer`

### CONTRIBUTING.md Structure
vLLM's `CONTRIBUTING.md` in the repo root is a **one-liner** pointing to their full docs site: `docs.vllm.ai/en/latest/contributing`. The actual contribution guide is comprehensive and hosted on their documentation site. Key sections:

1. **Ways to contribute**: Bug reports, model support, features, docs, community support, social media advocacy
2. **Job Board**: Curated lists of tasks — Good first issues, onboarding tasks, new model requests, multi-modal models
3. **License**: Apache-2.0
4. **Developing**: Detailed setup with `uv`, Python 3.12, Rust frontend builds, CUDA/C++ development
5. **Linting**: `pre-commit` hooks (auto-runs on commit)
6. **Documentation**: MkDocs-based with live preview
7. **Testing**: `pytest`-based
8. **Issues**: Search-first guidance, security disclosure path
9. **Pull Requests & Code Reviews**:
   - DCO/Signed-off-by requirement
   - **AI Assisted Contributions** policy (disclosure required, commit trailers like `Co-authored-by:`)
   - PR title classification system: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[Hardware][Vendor]`, `[Misc]`
   - Code quality standards (Google Python/C++ style guides)
   - Adding/changing kernels guidelines
   - Notes for large changes (>500 LOC requires RFC)
   - Transparent review process with SLAs (2-3 day status updates, 7-day escalation)
   - PR limits (6 open PRs for contributors without write access)
   - Expedited review via email with company/university verification

### Good First Issues
✅ Yes — actively labeled with `good first issue` and `help wanted`. Multiple open issues visible (10+). Issues span docs, kernels, refactoring, model support, and bug fixes. Many also tagged with `help wanted` to signal community contribution areas.

### Issue Templates
✅ Comprehensive — **9 templates** covering:
- `100-documentation.yml`
- `200-installation.yml`
- `300-usage.yml`
- `400-bug-report.yml`
- `450-ci-failure.yml`
- `500-feature-request.yml`
- `600-new-model.yml`
- `700-performance-discussion.yml`
- `750-RFC.yml`
- Plus a `config.yml`

### Community Channels
- **User Forum**: [discuss.vllm.ai](https://discuss.vllm.ai)
- **Developer Slack**: [slack.vllm.ai](https://slack.vllm.ai)
- **Twitter/X**: [@vllm_project](https://x.com/vllm_project)
- **GitHub Issues**: For technical questions and feature requests
- **GitHub Security Advisories**: For security disclosures
- **Email**: collaboration@vllm.ai (for partnerships)

### README Contributor Section
Brief: *"We welcome and value any contributions and collaborations. Please check out Contributing to vLLM for how to get involved."* — links to docs site. Also has a "Contact Us" section with clear routing (issues, forum, Slack, security, partnerships).

### Standout Practices
- **Job Board concept**: Curated task lists organized by difficulty/area — brilliant for onboarding
- **PR classification system**: Structured title prefixes make triage and review routing efficient
- **Transparent review SLAs**: Sets clear expectations (2-3 day updates, 7-day escalation)
- **AI policy with nuance**: Allows AI-generated code with disclosure requirements, bans trivial AI PRs
- **PR limits**: Prevents contributor overwhelm, ensures focus
- **Expedited review path**: For production/research-critical contributions
- **Multiple issue templates**: 9 specialized templates for precise routing
- **Pre-commit hooks**: Automated quality enforcement
- **Separate user/developer channels**: Forum for users, Slack for developers

---

## 2. SGLang

**Repository**: [sgl-project/sglang](https://github.com/sgl-project/sglang)  
**Stars**: 31k | **Forks**: 7.5k | **Commits**: 15,888

### GitHub Description
> "SGLang is a high-performance serving framework for large language models and multimodal models."

### Topics
`attention`, `blackwell`, `cuda`, `deepseek`, `diffusion`, `glm`, `gpt-oss`, `inference`, `llama`, `llm`, `minimax`, `moe`, `qwen`, `qwen-image`, `reinforcement-learning`, `transformer`, `vlm`, `wan`

### CONTRIBUTING.md Structure
SGLang does **not** have a `CONTRIBUTING.md` in the repo root. Instead, the contribution guide lives on their documentation site at `docs.sglang.io/developer_guide/contribution_guide.html`. Structure:

1. **Install from source**: Fork/clone, build from source
2. **Format code with pre-commit**: `pre-commit` installation and usage, CI-enforced link checking
3. **Run and add unit tests**:
   - Unit tests (no server required) under `test/registered/unit/` — organized to mirror source tree
   - E2E tests (server required)
   - Coverage with `--cov`
4. **Write documentation**: Recommended as a starting point for new contributors
5. **Test the accuracy**: GSM8K sanity checks, accuracy eval examples
6. **Benchmark the speed**: Links to profiling guide
7. **Requesting a review for merge**: Process with Merge Oncall, Codeowner, and reviewers
8. **How to trigger CI tests**:
   - Permission-gated CI triggering
   - Slash commands: `/tag-run-ci-label`, `/rerun-failed-ci`, `/tag-and-rerun-ci`, `/rerun-stage`, `/rerun-test`
   - CI rate limits with cooldown periods
9. **Code style guidance**:
   - Avoid code duplication
   - Minimize device synchronization
   - Extreme efficiency (runtime is on critical path)
   - Pure functions preferred
   - File size limits (2,000 lines)
   - Test speed requirements
   - Security: Never use `pickle.loads()` for untrusted data
   - Hardware support guidelines (don't change existing code, prefer new files)
10. **How to update sgl-kernel**: Multi-PR process for kernel changes
11. **Tips for newcomers**:
    - Good first issue and help wanted labels
    - [Mini-SGLang](https://github.com/sgl-project/mini-sglang) — quick overview of codebase structure
    - Code Walk-through materials
    - GTC-2026 Training Lab for hands-on practice

### Good First Issues
✅ Yes — labeled with `good first issue` and `help wanted`. Contribution guide explicitly directs newcomers to these.

### Issue Templates
Unable to fetch the issue template directory (network error), but the contribution guide references structured CI workflows and PR processes.

### Community Channels
- **Slack**: [slack.sglang.io](https://slack.sglang.io)
- **Weekly Dev Meeting**: [meet.sglang.io](https://meet.sglang.io)
- **Blog**: [lmsys.org/blog/](https://lmsys.org/blog/)
- **Roadmap**: [roadmap.sglang.io](https://roadmap.sglang.io)
- **Learning Materials**: [sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)

### README Contributor Section
The README links to the contribution guide at `docs.sglang.io/developer_guide/contribution_guide.html` under "Getting Started". Also mentions: "Long-term active SGLang contributors are eligible for coding agent sponsorship, such as Cursor, Claude Code, or OpenAI Codex."

### Standout Practices
- **Mini-SGLang**: A dedicated simplified codebase for learning the architecture — exceptional onboarding tool
- **Weekly dev meetings**: Regular synchronous coordination open to community
- **Public roadmap**: Transparent roadmap at roadmap.sglang.io
- **Contributor rewards**: Coding agent sponsorship (Cursor, Claude Code, Codex) for active contributors
- **Slash-command CI system**: Powerful `/rerun-failed-ci`, `/rerun-stage`, `/rerun-test` commands for granular CI control
- **Documentation as onboarding**: Recommends new contributors start with docs
- **Code walk-through resources**: Structured learning materials for deep codebase understanding
- **GTC Training Lab**: Hands-on optimization and benchmarking workshops
- **Multi-PR process**: Kernel changes require staged PRs — maintains quality

---

## 3. Exo

**Repository**: [exo-explore/exo](https://github.com/exo-explore/exo)  
**Stars**: 46.5k | **Forks**: 3.4k | **Commits**: 2,353

### GitHub Description
> "Run frontier AI locally."

### Topics
None listed on the repo page.

### CONTRIBUTING.md Structure
Exo has a dedicated `CONTRIBUTING.md` in the repo root. It's **moderately detailed** and well-structured:

1. **Getting Started**: Prerequisites (uv, Rust nightly, macmon), clone and run instructions
2. **Development**: 
   - Tech stack: Rust, Python, TypeScript (Svelte)
   - Keep changes focused — one feature/fix per PR
   - Pull latest source before starting
3. **Code Style**: 
   - Pure functions where possible
   - Prefer Rust for new code
   - Leverage type systems (Rust, Python type hints, TypeScript types)
   - Comments explain "why" not "what"
   - Auto-format with `nix fmt`
4. **Model Cards**: Detailed guide for TOML-based model card format
   - Required/optional fields documented
   - Security note on `trust_remote_code`
5. **API Adapters**: Architecture pattern for adding new API format support
   - Adapter pattern with conversion functions
   - Step-by-step guide for adding new adapters
   - Clear boundary between adapter and core systems
6. **Testing**: "Relies heavily on manual testing" — acknowledges limitation, invites automated tests
7. **Submitting Changes**: Standard fork/branch/PR workflow
8. **Reporting Issues**: Clear expectations (description, steps, environment)
9. **Questions?**: Links to X/Twitter

### Good First Issues
Not explicitly found on the GitHub page (no topics listed, less issue labeling infrastructure visible).

### Issue Templates
Unable to fetch (network error), but the contributing guide describes issue reporting expectations.

### Community Channels
- **Discord**: [discord.gg/TJ4P57arEm](https://discord.gg/TJ4P57arEm)
- **X/Twitter**: [@exolabs](https://x.com/exolabs)

### README Contributor Section
Brief: *"See CONTRIBUTING.md for guidelines on how to contribute to exo."*

### Standout Practices
- **Model card system**: TOML-based model cards are a clear extension point for contributors — add new models by creating a simple config file
- **API adapter architecture**: Well-documented pattern for extending API compatibility — clear, isolated extension point
- **Honest about limitations**: Acknowledges heavy reliance on manual testing while actively working to improve
- **Multi-language guidance**: Specific guidance for the Rust/Python/TypeScript stack
- **Type system emphasis**: Encourages leveraging type systems across all three languages

---

## 4. Petals

**Repository**: [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)  
**Stars**: 10.5k | **Forks**: 636 | **Commits**: 522

### GitHub Description
> "🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading"

### Topics
`bloom`, `chatbot`, `deep-learning`, `distributed-systems`, `falcon`, `gpt`, `guanaco`, `language-models`, `large-language-models`, `llama`, `machine-learning`, `mixtral`, `neural-networks`, `nlp`, `pipeline-parallelism`, `pretrained-models`, `pytorch`, `tensor-parallelism`, `transformer`, `volunteer-computing`

### CONTRIBUTING.md Structure
Petals has **no `CONTRIBUTING.md` in the repo root**. Contributing guidance is in the README ("Please see our FAQ on contributing") and detailed in the **GitHub Wiki FAQ**. The FAQ's "Contributing" section covers:

1. **How to help**: Check issues with `good first issue` and `help wanted` tags
2. **Good first issue criteria**: Can be solved in 1-2 days, provides good codebase study opportunity
3. **Collaboration invitation**: After addressing some issues, team is happy to collaborate on more impactful tasks
4. **Style guide**: `black` and `isort` for all PRs
5. **Testing**:
   - Small changes: Draft PR for CI testing
   - Larger changes: Private swarm testing with instructions for setup
   - Specific commands provided for running servers and tests

### Good First Issues
✅ Yes — labeled with `good first issue` and `help wanted`. The FAQ explicitly defines what makes a good first issue (1-2 days, codebase study opportunity).

### Issue Templates
Not visible from the data gathered. The repo appears to have minimal issue template infrastructure (only 522 commits, smaller team).

### Community Channels
- **Discord**: [discord.gg/KdThf2bWVU](https://discord.gg/KdThf2bWVU) (multiple specialized channels)
  - `#discussion` — general questions
  - `#dev` — development questions
  - `#running-a-client` — client help
  - `#running-a-server` — server help
- **Swarm Monitor**: [health.petals.dev](https://health.petals.dev) — real-time swarm status
- **Chat App**: [chat.petals.dev](https://chat.petals.dev) — demo app

### README Contributor Section
Brief: *"Please see our FAQ on contributing."* — links to wiki. Also has a prominent section "Connect your GPU and increase Petals capacity" that frames **running a server as a form of contribution**.

### Standout Practices
- **Contribution beyond code**: Contributing GPU time to the swarm is explicitly valued — "connect your GPU" is a form of contribution
- **Recognition system**: Top contributors get names/links displayed on the swarm monitor
- **Specialized Discord channels**: Separate channels for different contributor needs (dev, client, server)
- **Clear good first issue criteria**: Defines exactly what makes an issue suitable for newcomers
- **Private swarm testing**: Detailed instructions for testing server changes without affecting public swarm
- **Low barrier for model hosting**: Hosting model layers is itself a meaningful contribution
- **Research paper integration**: Academic credibility through published papers

---

## 5. llama.cpp

**Repository**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
**Stars**: 122k | **Forks**: 21.2k | **Commits**: 10,205

### GitHub Description
> "LLM inference in C/C++"

### Topics
`ggml` (single topic — minimal tagging)

### CONTRIBUTING.md Structure
llama.cpp has the **most comprehensive CONTRIBUTING.md** of all projects analyzed. It covers:

1. **Contributor Levels**:
   - **Contributors**: No special privileges
   - **Collaborators (Triage)**: Significant contributions, responsible for parts of code, maintain and review
   - **Maintainers**: Review and merge PRs after approval from code owners

2. **AI Usage Policy** (prominent, with `> [!IMPORTANT]` callout):
   - AI-generated code is allowed, but you're 100% responsible
   - **Undisclosed AI usage may result in permanent ban**
   - Must disclose how AI was used
   - Must check for duplicate PRs
   - Must perform comprehensive manual review
   - Must explain every line when asked
   - **Strictly prohibited to use AI for bug reports, feature requests, PR descriptions, discussions, or human responses**
   - References separate `AGENTS.md` file

3. **Pull Requests (for contributors & collaborators)**:
   - **Before you start**:
     - Search for existing discussions/PRs
     - Features must begin with an issue, not a PR
     - Bug-fix PRs must include reproducible issue and regression test
     - New CLI/API additions have higher bar
     - New contributors: limit to 1 open PR, no trivial fixes
   - **Preparing your PR**:
     - Familiarize with ggml tensor library
     - Execute full CI locally
     - Verify perplexity and performance not negatively affected
     - Run `test-backend-ops` for ggml modifications
     - Separate PRs for each feature
     - CPU support first, other backends in follow-up PRs
     - New quantization types: provide GGUF conversion, perplexity comparisons, KL divergence data, performance benchmarks
   - **After submitting**:
     - Expect modification requests
     - Rebase stale PRs
     - Consider adding yourself to CODEOWNERS

4. **Pull Requests (for maintainers)**:
   - Squash-merge format: `<module> : <commit title> (#<issue_number>)`
   - Let other maintainers merge their own PRs
   - `[no release]` tag for non-release changes
   - "merge ready" label for fast-merging
   - Right to decline/close PRs

5. **Coding Guidelines**:
   - Avoid third-party dependencies
   - Cross-compatibility considerations
   - Avoid modern STL constructs, keep it simple
   - Vertical alignment for readability
   - 4 spaces indentation, specific bracket style
   - Sized integer types in public API
   - Specific struct/enum declaration style
   - `clang-format` for formatting
   - C++ Core Guidelines reference
   - Matrix multiplication convention (unconventional, documented)

6. **Naming Guidelines**:
   - `snake_case` for everything
   - Longest common prefix optimization
   - Enum values uppercase, prefixed with enum name
   - `<class>_<method>` pattern
   - `_t` suffix for opaque types
   - Specific file naming conventions

7. **Code Maintenance**:
   - CODEOWNERS-based ownership
   - CI workflow responsibility
   - Server development documentation reference

8. **Documentation**: Community effort, encourage adding summaries to headers

9. **Resources**: Links to GitHub projects page

### Good First Issues
✅ Yes — 12+ open issues with `good first issue` label. Also uses `help wanted`. Issues range from eval bugs, feature requests, GGUF conversion, TTS support, compile bugs, to structured output improvements.

### Issue Templates
✅ **6 templates** covering:
- `010-bug-compilation.yml`
- `011-bug-results.yml`
- `019-bug-misc.yml`
- `020-enhancement.yml`
- `030-research.yml`
- `040-refactor.yml`
- Plus `config.yml`

### Community Channels
- **GitHub Discussions**: Used for community Q&A
- **GitHub Projects**: Curated resource lists
- **Wiki**: GGML Tips & Tricks
- No Discord/Slack found — primarily GitHub-centric

### README Contributor Section
Detailed section with contributor hierarchy:
- "Contributors can open PRs"
- "Collaborators will be invited based on contributions"
- "Maintainers can push to branches and merge PRs"
- "Any help with managing issues, PRs and projects is very appreciated"
- Links to CONTRIBUTING.md

### Standout Practices
- **Most rigorous AI policy**: Permanent ban for undisclosed AI usage — sets a strong quality bar
- **Contributor levels**: Clear hierarchy (Contributor → Collaborator → Maintainer) with defined privileges
- **No trivial fixes rule**: New contributors limited to 1 PR and no trivial fixes — prevents noise
- **Features require issues first**: Prevents wasted effort on unwanted features
- **Regression test requirement**: Bug fixes must include test that fails before and passes after
- **Naming conventions**: Extremely detailed naming guidelines ensure consistency
- **New contributor limits**: 1 open PR max for newcomers
- **CODEOWNERS**: Explicit code ownership for review routing
- **Manifesto link**: Links to a philosophical document about the project's direction
- **Merge ready label**: Fast-track for simple PRs that don't need 2 reviews

---

## 6. Ollama

**Repository**: [ollama/ollama](https://github.com/ollama/ollama)  
**Stars**: 177k | **Forks**: 17.2k | **Commits**: 5,586

### GitHub Description
> "Get up and running with Kimi-K2.6, GLM-5.2, MiniMax, DeepSeek, gpt-oss, Qwen, Gemma and other models."

### Topics
`deepseek`, `gemma`, `gemma3`, `glm`, `go`, `golang`, `gpt-oss`, `llama`, `llama3`, `llm`, `llms`, `minimax`, `mistral`, `ollama`, `qwen`

### CONTRIBUTING.md Structure
Ollama has a **well-structured, concise `CONTRIBUTING.md`** in the repo root:

1. **Setup**: Links to development docs (`docs/development.md`)
2. **Ideal Issues** (categorized by ease of acceptance):
   - **Bugs**: Where Ollama stops working or shows unexpected errors
   - **Performance**: Inference, downloading, uploading speed
   - **Security**: Per SECURITY.md, no public disclosure
3. **Harder to Review**:
   - New features (API fields, environment variables) — add maintenance burden
   - Refactoring — important but slower to review
   - Documentation — small updates welcome, large additions hard to maintain
4. **May Not Be Accepted**:
   - Breaking backward compatibility
   - Significant UX friction
   - Large future maintenance burden
5. **Proposing Non-Trivial Changes**:
   - Must open an issue first to discuss
   - Tips: explain the problem, importance, usage, testing
   - Bonus: provide draft documentation
6. **Pull Requests**:
   - **Commit messages**: `<package>: <short description>` format, lowercase, continuation of "This changes Ollama to..."
   - **Tests**: Include tests, test behavior not implementation
   - **New dependencies**: Added sparingly, must justify
7. **Need Help?**: Discord server

### Good First Issues
✅ Yes — 3 open issues with `good first issue` label (fewer than other projects). Issues are specific: sidebar animation bug, CLI image path bug, Windows sleep prevention.

### Issue Templates
✅ **3 templates**:
- `10_bug_report.yml` — Bug reports (YAML form)
- `20_feature_request.md` — Feature requests (Markdown)
- `30_model_request.md` — Model requests (Markdown)
- Plus `config.yml`

### Community Channels
- **Discord**: [discord.gg/ollama](https://discord.gg/ollama)
- **Twitter/X**: [@ollama](https://x.com/ollama)
- **Reddit**: [r/ollama](https://reddit.com/r/ollama)

### README Contributor Section
The README focuses on users, not contributors. Contributing info is in the sidebar link to `CONTRIBUTING.md`. The README does have a massive "Community Integrations" section listing 100+ integrations — showing the ecosystem strength.

### Standout Practices
- **Acceptance categorization**: Clearly triages issues into "ideal," "harder to review," and "may not be accepted" — manages contributor expectations brilliantly
- **Require issue before feature PR**: Prevents wasted effort
- **Commit message format**: `<package>: <description>` with semantic convention ("This changes Ollama to...")
- **Dependency skepticism**: New deps must be justified
- **Problem-first proposals**: Tips to explain the problem, not the solution
- **Draft documentation bonus**: Encourages thinking about user-facing impact
- **Massive integration ecosystem**: 100+ community integrations show how a simple, well-documented API creates an ecosystem
- **Multi-platform community**: Discord, Reddit, and Twitter for different engagement styles

---

## 7. Best Practices Summary & Recommendations for Synapse

### What Makes Developers Want to Contribute

Based on analyzing these 6 projects, the key factors that drive contribution are:

#### 1. **Low Barrier to Entry**
- **Quick start in < 5 minutes**: vLLM, SGLang, and Exo all prioritize fast setup
- **Pre-commit hooks**: Automated quality checks reduce cognitive load (vLLM, SGLang, llama.cpp)
- **Documentation as entry point**: SGLang recommends docs as the first contribution — lowers barrier significantly
- **Model cards/config files**: Exo's TOML model cards and Ollama's model requests let non-developers contribute

#### 2. **Clear Task Discovery**
- **Curated task lists**: vLLM's "Job Board" concept (good first issues, onboarding tasks, model requests) is the gold standard
- **Good first issues with criteria**: Petals defines exactly what makes a good first issue (1-2 days, learning opportunity)
- **Acceptance categorization**: Ollama's triage into "ideal/harder/may not accept" prevents wasted effort
- **"help wanted" labels**: Used across all projects to signal community-welcome areas

#### 3. **Transparent Process**
- **Review SLAs**: vLLM's 2-3 day update cadence and 7-day escalation
- **Public roadmaps**: SGLang's public roadmap shows direction and where help is needed
- **Weekly dev meetings**: SGLang's open meetings build community
- **Clear PR expectations**: llama.cpp's detailed before/during/after PR guidance

#### 4. **Recognition & Reward**
- **Contributor levels**: llama.cpp's Contributor → Collaborator → Maintainer progression
- **Public recognition**: Petals shows contributor names on swarm monitor
- **Tangible rewards**: SGLang's coding agent sponsorship for active contributors
- **CODEOWNERS**: llama.cpp's code ownership gives contributors responsibility and recognition

#### 5. **Quality Without Gatekeeping**
- **AI policies**: Both vLLM and llama.cpp address AI-generated code transparently
- **Regression tests required**: llama.cpp requires bug fixes to include failing-then-passing tests
- **New contributor limits**: llama.cpp's 1-PR limit for newcomers prevents overwhelm
- **No trivial fixes**: llama.cpp discourages one-line typo PRs from new contributors

#### 6. **Community Infrastructure**
- **Separate channels for different needs**: vLLM (forum for users, Slack for devs), Petals (Discord channels for dev/client/server)
- **Multiple community platforms**: Ollama's Discord + Reddit + Twitter covers different demographics
- **Swarm/community health dashboards**: Petals' health monitor, SGLang's roadmap site

---

### Actionable Recommendations for Synapse

#### Priority 1: Foundation (Do First)

1. **Write a comprehensive CONTRIBUTING.md** modeled on Ollama's clarity + vLLM's completeness:
   - Categorize contribution types by acceptance likelihood
   - Include setup instructions (aim for < 5 min quick start)
   - Define commit message format
   - Link to a development guide

2. **Create issue templates** (minimum 3):
   - Bug report (YAML form — vLLM/llama.cpp style)
   - Feature request
   - Model/request template (if applicable)
   - Use a `config.yml` to configure the issue template chooser

3. **Label good first issues** — aim for 5-10 at any time:
   - Define criteria (Petals model: 1-2 days, learning opportunity)
   - Pair with `help wanted` for broader community tasks
   - Create a "Job Board" page or README section linking to curated tasks

4. **Set up pre-commit hooks**: Automated linting/formatting reduces review friction

#### Priority 2: Community (Do Next)

5. **Choose primary community channel** (Discord recommended based on ecosystem):
   - Create specialized channels: `#general`, `#dev`, `#help`, `#feature-ideas`
   - Consider a weekly/bi-weekly dev sync (SGLang model)

6. **Write AI contribution policy**: Be explicit about AI-generated code expectations (llama.cpp/vLLM model)

7. **Create a project roadmap**: Public visibility into direction (SGLang model — even a simple GitHub Projects board)

#### Priority 3: Growth (Scale Up)

8. **Build a "Mini-Synapse"**: Simplified codebase walkthrough (SGLang's mini-sglang is exceptional for onboarding)

9. **Define contributor levels**: Contributor → Collaborator → Maintainer with clear privileges (llama.cpp model)

10. **Consider contributor rewards**: Coding agent sponsorship, swag, or recognition systems

11. **Document architecture extension points**: Like Exo's API adapters or model cards — clear patterns for extending the system

12. **Separate user/developer channels**: Forum or Discord for users, Slack/GitHub Discussions for developers

---

### Comparative Matrix

| Feature | vLLM | SGLang | Exo | Petals | llama.cpp | Ollama |
|---------|------|--------|-----|--------|-----------|--------|
| **Stars** | 87.8k | 31k | 46.5k | 10.5k | 122k | 177k |
| **CONTRIBUTING.md** | Links to docs site | Links to docs site | In-repo, moderate | Wiki FAQ | In-repo, comprehensive | In-repo, concise |
| **Good First Issues** | ✅ Many | ✅ Yes | ❌ Limited | ✅ Yes | ✅ Many | ✅ Few |
| **Issue Templates** | 9 templates | Unknown | Unknown | Minimal | 6 templates | 3 templates |
| **AI Policy** | ✅ Detailed | ❌ | ❌ | ❌ | ✅ Strictest | ❌ |
| **Contributor Levels** | ❌ | ❌ | ❌ | ❌ | ✅ 3-level | ❌ |
| **PR Review SLAs** | ✅ 2-3 days | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Community Channel** | Forum + Slack | Slack | Discord | Discord | GitHub-centric | Discord + Reddit |
| **Public Roadmap** | ❌ | ✅ | ❌ | ❌ | ✅ Projects | ❌ |
| **Pre-commit Hooks** | ✅ | ✅ | ✅ (nix fmt) | ✅ (black/isort) | ✅ | ❌ |
| **CODEOWNERS** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Learning Resources** | ❌ | ✅ Mini-SGLang | ❌ | ❌ | ✅ Wiki | ❌ |

---

### Key Takeaway

The most compelling contributor experiences share these traits:

1. **Make the first contribution easy**: Quick setup, clear tasks, documentation as entry point
2. **Set expectations clearly**: What will/won't be accepted, review timelines, quality standards
3. **Respect contributor time**: Good issue labeling, no-waste-before-you-start rules (discuss first)
4. **Build community, not just code**: Regular meetings, recognition systems, specialized channels
5. **Automate quality**: Pre-commit hooks, CI systems, slash commands for re-running tests
6. **Create extension points**: Model cards, API adapters, plugin systems that invite contribution

The best projects treat contributor experience as a product — they design it intentionally, iterate on it, and measure its effectiveness through contributor retention and code quality.
