Proposed agent roles
1. Research Librarian Agent

This agent reads papers, blog posts, docs, and notes, then maps them to your existing infrastructure.

Its job is not to invent alpha directly. Its job is to answer:

what is the paper’s core claim?

what features does it imply?

what assumptions does it make?

which part of your stack can implement it?

For your setup, it should relate papers to things like:

entropy.rs

orderbook.rs

labeler.rs

ml.rs

OMS.rs

Example output:

{
  "paper_id": "arxiv_2026_xxx",
  "title": "Entropy-Based Limit Order Book Signals",
  "core_claim": "Short-horizon order flow entropy predicts reversal probability",
  "related_modules": ["entropy.rs", "labeler.rs"],
  "candidate_features": ["tick_entropy_1s", "vol_tick_entropy_5s"],
  "testable_hypotheses": [
    "high tick entropy predicts weaker continuation over 30s"
  ]
}

This agent is very valuable because it turns literature into implementation tasks.

2. Hypothesis Agent

This is the scientific brain of the system.

It takes inputs from:

research papers

your prior experiments

your own ideas

market observations

and produces clean, testable hypotheses.

A good hypothesis must be:

measurable

falsifiable

tied to a horizon

tied to a market

tied to a feature family

Good example:

In BTC perpetuals on 100ms data, elevated 5s volume tick entropy combined with high PWI asymmetry predicts mean reversion over the next 30–60 seconds.

Bad example:

Entropy probably matters in markets.

This agent should never write code directly. It should produce structured experiment proposals.

3. Experiment Designer Agent

This agent converts a hypothesis into a rigorous experiment spec.

It defines:

market

timeframe

data slice

features

labels

model class

baseline

metrics

transaction cost assumptions

acceptance criteria

Example:

{
  "experiment_id": "exp_0142",
  "hypothesis": "High entropy predicts mean reversion",
  "market": "BTCUSDT_PERP",
  "data_frequency": "100ms",
  "features": ["tick_entropy_1s", "tick_entropy_5s", "pwi_25", "microprice_dev"],
  "label": "3_bar_directional",
  "model": "svm_rbf",
  "baseline": "microprice_only",
  "metrics": ["auc", "f1", "sharpe_net", "turnover", "max_drawdown"],
  "cost_model": {
    "fees_bps": 5,
    "slippage_bps": 2
  },
  "acceptance": {
    "min_sharpe_net": 1.0,
    "min_auc": 0.54
  }
}

This agent is crucial because most quant errors happen before backtesting even starts.

4. Code Builder Agent

This agent writes or modifies code to implement the experiment.

It should operate with strong constraints:

only edit allowed modules

always generate tests

never deploy directly

produce diffs and explanations

link code changes to experiment IDs

It might create:

a new feature in entropy.rs

a label variation in labeler.rs

a config entry in experiments/

a Python or Rust backtest adapter

unit tests and integration tests

For you, this is where Claude Code or a similar coding agent becomes very useful.

But do not let this agent have unchecked authority. Every code change should be traceable.

5. Backtest Agent

This agent runs the experiment.

It should be mostly deterministic, even if wrapped by an AI coordinator.

Its responsibilities:

fetch the right dataset

compute features

run the model or rule

apply costs

store artifacts

log results

Artifacts should include:

config used

code version / git commit

train/test split

metrics

equity curve

confusion matrix

feature importance

diagnostic plots

This is one of the most important parts for credibility. Investors and hiring managers will care far more about rigorous backtests than fancy agent chatter.

6. Critic / Validation Agent

This is the skeptic.

Its job is to attack results and reduce false positives.

It asks questions like:

is this overfit?

is the baseline too weak?

is the split leaking information?

are costs realistic?

does performance survive another regime?

is there multiple-testing bias?

is the signal economically meaningful?

This agent is essential. Without it, your lab will produce beautiful nonsense.

Example output:

{
  "experiment_id": "exp_0142",
  "status": "challenged",
  "issues": [
    "performance concentrated in one month only",
    "turnover too high after realistic costs",
    "label leakage suspected via window overlap"
  ],
  "recommended_actions": [
    "rerun with embargoed split",
    "test on ETH and AVAX",
    "increase slippage assumptions"
  ]
}

This is where your lab starts to look professional.

7. Reporter Agent

This agent writes the human-facing summary and publishes it to your website/dashboard.

It produces three versions:

Internal technical report

For you.

Contains:

full hypothesis

implementation notes

detailed metrics

code commit

validation notes

next steps

Investor-facing report

For external visibility.

Contains:

research progress

transparency

selected metrics

methodology overview

activity logs

no exaggerated claims

Recruiter / academic report

For quant jobs or PhD applications.

Contains:

research question

methodology

engineering architecture

reproducibility

originality

limitations

This agent turns your work into narrative capital.

Orchestrator role

Above these, you need one Supervisor / Orchestrator.

This is not really a creative agent. It is the lab manager.

It decides:

which agent runs next

when human approval is required

where artifacts are stored

how runs are retried

how failures are handled

Flow could be:

Research Librarian
   → Hypothesis Agent
   → Experiment Designer
   → Code Builder
   → Backtest Agent
   → Critic Agent
   → Reporter Agent

The orchestrator also attaches:

run_id

experiment_id

timestamps

git commit

model version

dataset version

Without this layer, the lab becomes messy quickly.

Deterministic services around the agents

These should not be agents.

They should be normal services:

market data ingestion

feature calculation runtime

backtest engine

database

artifact store

dashboard backend

auth and permissions

queue / job execution

This matters a lot. Your actual edge will come from deterministic infrastructure plus good hypotheses, not from agent theatrics.

What should be visible on the website

Your website should make the lab legible.

I would expose these pages:

Experiments

List of all hypotheses, status, metrics, validation outcome

Agents

Current activity of each agent, last action, errors, duration

Live logs

Structured log stream by experiment and agent

Research reports

Readable summaries with charts and conclusions

Strategy candidates

Only experiments that passed validation gates

Infrastructure health

Data freshness, queue backlog, failed jobs, code version

This is powerful because it makes your work visible to:

you

collaborators

recruiters

investors

Logging structure

Every agent should emit structured events, not plain text spam.

Use fields like:

{
  "timestamp": "2026-03-10T19:45:00Z",
  "run_id": "run_20260310_001",
  "experiment_id": "exp_0142",
  "agent": "critic_agent",
  "event_type": "validation_warning",
  "level": "WARN",
  "message": "Sharpe collapses after higher slippage assumptions",
  "metadata": {
    "old_sharpe": 1.42,
    "new_sharpe": 0.38
  }
}

Send these by HTTP to a backend, and use WebSocket from backend to browser for live viewing.

Best initial version for you

Given your background, I would not start with all 7 at full power.

I would start with this minimal set:

Research Librarian Agent

Hypothesis Agent

Experiment Designer Agent

Backtest Agent

Reporter Agent

And keep these mostly deterministic:

Code Builder

Critic

Because early on, you want rigor more than autonomy.

Why this could help your career

This architecture is strong because it demonstrates four things at once:

quant research discipline

systems engineering ability

AI orchestration skill

reproducible experimental thinking

That combination is rare.

For a quant job, this can be very compelling if you show:

serious backtests

careful skepticism

transparent logs

strong implementation detail

For a PhD, it becomes interesting if you emphasize:

automated scientific workflow

entropy/microstructure novelty

experiment reproducibility

agent-assisted hypothesis generation

So yes, this can become more than a side project. It can become a portfolio centerpiece.

My blunt assessment

This is a good idea, but only if you frame it correctly.

Do not frame it as:

a bunch of AI bots making money

Frame it as:

an AI-assisted quantitative research laboratory with reproducible experimentation, transparent logging, and disciplined validation

That sounds more serious because it is more serious.

Next, I can turn this into a concrete repo structure and service architecture so you have:

folders

agent boundaries

APIs

database tables

dashboard pages
