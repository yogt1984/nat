"""Tests for the unified nat CLI argument parsing and dispatch."""

import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

# Import nat module (no .py extension — need explicit loader)
_nat_path = str(ROOT / "nat")
_loader = importlib.machinery.SourceFileLoader("nat_cli", _nat_path)
_spec = importlib.util.spec_from_file_location("nat_cli", _nat_path, loader=_loader)
nat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nat)


@pytest.fixture
def parser():
    return nat.build_parser()


# ── Top-level commands ───────────────────────────────────────────────────────

class TestTopLevel:
    def test_no_args(self, parser):
        args = parser.parse_args([])
        assert args.command is None

    def test_help(self, parser):
        args = parser.parse_args(['help'])
        assert args.command == 'help'

    def test_start(self, parser):
        args = parser.parse_args(['start'])
        assert args.func == nat.cmd_start

    def test_stop(self, parser):
        args = parser.parse_args(['stop'])
        assert args.func == nat.cmd_stop

    def test_status(self, parser):
        args = parser.parse_args(['status'])
        assert args.func == nat.cmd_status

    def test_log(self, parser):
        args = parser.parse_args(['log'])
        assert args.func == nat.cmd_log

    def test_profile(self, parser):
        args = parser.parse_args(['profile'])
        assert args.func == nat.cmd_profile

    def test_report(self, parser):
        args = parser.parse_args(['report'])
        assert args.func == nat.cmd_report

    def test_dashboard(self, parser):
        args = parser.parse_args(['dashboard'])
        assert args.func == nat.cmd_dashboard

    def test_macro(self, parser):
        args = parser.parse_args(['macro'])
        assert args.func == nat.cmd_macro

    def test_screen(self, parser):
        args = parser.parse_args(['screen'])
        assert args.func == nat.cmd_screen

    def test_screen_with_positional_args(self, parser):
        args = parser.parse_args(['screen', 'BTC', 'ETH'])
        assert args.func == nat.cmd_screen
        assert 'BTC' in args.screen_args


# ── Build commands ───────────────────────────────────────────────────────────

class TestBuild:
    def test_build_default_is_release(self, parser):
        args = parser.parse_args(['build'])
        assert args.func == nat.cmd_build

    def test_build_debug(self, parser):
        args = parser.parse_args(['build', 'debug'])
        assert args.func == nat.cmd_build_debug

    def test_build_api(self, parser):
        args = parser.parse_args(['build', 'api'])
        assert args.func == nat.cmd_build_api

    def test_build_clean(self, parser):
        args = parser.parse_args(['build', 'clean'])
        assert args.func == nat.cmd_build_clean

    def test_build_fmt(self, parser):
        args = parser.parse_args(['build', 'fmt'])
        assert args.func == nat.cmd_build_fmt

    def test_build_lint(self, parser):
        args = parser.parse_args(['build', 'lint'])
        assert args.func == nat.cmd_build_lint

    def test_build_check(self, parser):
        args = parser.parse_args(['build', 'check'])
        assert args.func == nat.cmd_build_check


# ── Run commands ─────────────────────────────────────────────────────────────

class TestRun:
    def test_run_default(self, parser):
        args = parser.parse_args(['run'])
        assert args.func == nat.cmd_run

    def test_run_serve(self, parser):
        args = parser.parse_args(['run', 'serve'])
        assert args.func == nat.cmd_run_serve

    def test_run_show_defaults(self, parser):
        args = parser.parse_args(['run', 'show'])
        assert args.func == nat.cmd_run_show
        assert args.symbol == 'BTC'
        assert args.freq == 1

    def test_run_show_custom(self, parser):
        args = parser.parse_args(['run', 'show', '--symbol', 'ETH', '--freq', '10'])
        assert args.func == nat.cmd_run_show
        assert args.symbol == 'ETH'
        assert args.freq == 10

    def test_run_tunnel(self, parser):
        args = parser.parse_args(['run', 'tunnel'])
        assert args.func == nat.cmd_run_tunnel


# ── Data commands ────────────────────────────────────────────────────────────

class TestData:
    def test_data_default(self, parser):
        args = parser.parse_args(['data'])
        assert args.func == nat.cmd_data

    def test_data_validate(self, parser):
        args = parser.parse_args(['data', 'validate'])
        assert args.func == nat.cmd_data_validate
        assert args.hours is None

    def test_data_validate_hours(self, parser):
        args = parser.parse_args(['data', 'validate', '--hours', '12'])
        assert args.func == nat.cmd_data_validate
        assert args.hours == 12

    def test_data_explore(self, parser):
        args = parser.parse_args(['data', 'explore'])
        assert args.func == nat.cmd_data_explore

    def test_data_schema(self, parser):
        args = parser.parse_args(['data', 'schema'])
        assert args.func == nat.cmd_data_schema


# ── Test commands ────────────────────────────────────────────────────────────

class TestTestCmd:
    def test_test_default(self, parser):
        args = parser.parse_args(['test'])
        assert args.func == nat.cmd_test

    def test_test_unit(self, parser):
        args = parser.parse_args(['test', 'unit'])
        assert args.func == nat.cmd_test

    def test_test_verbose(self, parser):
        args = parser.parse_args(['test', 'verbose'])
        assert args.func == nat.cmd_test_verbose

    def test_test_hypotheses(self, parser):
        args = parser.parse_args(['test', 'hypotheses'])
        assert args.func == nat.cmd_test_hypotheses

    def test_test_validate_all(self, parser):
        args = parser.parse_args(['test', 'validate'])
        assert args.func == nat.cmd_test_validate
        assert args.target is None

    def test_test_validate_specific(self, parser):
        for target in ['api', 'positions', 'whales', 'entropy']:
            args = parser.parse_args(['test', 'validate', target])
            assert args.func == nat.cmd_test_validate
            assert args.target == target

    def test_test_backtest_coverage(self, parser):
        args = parser.parse_args(['test', 'backtest', '--coverage'])
        assert args.func == nat.cmd_test_backtest
        assert args.coverage is True

    def test_test_eamm(self, parser):
        args = parser.parse_args(['test', 'eamm'])
        assert args.func == nat.cmd_test_eamm
        assert args.integration is False

    def test_test_eamm_integration(self, parser):
        args = parser.parse_args(['test', 'eamm', '--integration'])
        assert args.func == nat.cmd_test_eamm
        assert args.integration is True

    def test_test_pipeline(self, parser):
        args = parser.parse_args(['test', 'pipeline'])
        assert args.func == nat.cmd_test_pipeline

    def test_test_pipeline_runner(self, parser):
        args = parser.parse_args(['test', 'pipeline-runner'])
        assert args.func == nat.cmd_test_pipeline_runner

    def test_test_dashboard(self, parser):
        args = parser.parse_args(['test', 'dashboard'])
        assert args.func == nat.cmd_test_dashboard

    def test_test_serving(self, parser):
        args = parser.parse_args(['test', 'serving'])
        assert args.func == nat.cmd_test_serving

    def test_test_api(self, parser):
        args = parser.parse_args(['test', 'api'])
        assert args.func == nat.cmd_test_api

    def test_test_redis(self, parser):
        args = parser.parse_args(['test', 'redis'])
        assert args.func == nat.cmd_test_redis

    def test_test_integration(self, parser):
        args = parser.parse_args(['test', 'integration'])
        assert args.func == nat.cmd_test_integration


# ── Backtest commands ────────────────────────────────────────────────────────

class TestBacktest:
    def test_backtest_default(self, parser):
        args = parser.parse_args(['backtest'])
        assert args.func == nat.cmd_backtest
        assert args.strategy == 'whale_flow_simple'
        assert args.symbol == 'BTC'

    def test_backtest_custom(self, parser):
        args = parser.parse_args(['backtest', '--strategy', 'momentum', '--symbol', 'ETH'])
        assert args.func == nat.cmd_backtest
        assert args.strategy == 'momentum'
        assert args.symbol == 'ETH'

    def test_backtest_validate(self, parser):
        args = parser.parse_args(['backtest', 'validate'])
        assert args.func == nat.cmd_backtest_validate

    def test_backtest_ml(self, parser):
        args = parser.parse_args(['backtest', 'ml', '--predictions', 'p.parquet', '--entry', '0.002'])
        assert args.func == nat.cmd_backtest_ml
        assert args.predictions == 'p.parquet'
        assert args.entry == 0.002

    def test_backtest_ml_validate(self, parser):
        args = parser.parse_args(['backtest', 'ml-validate'])
        assert args.func == nat.cmd_backtest_ml_validate

    def test_backtest_ml_quantile(self, parser):
        args = parser.parse_args(['backtest', 'ml-quantile', '--entry-q', '0.8'])
        assert args.func == nat.cmd_backtest_ml_quantile
        assert args.entry_q == 0.8

    def test_backtest_ml_tracked(self, parser):
        args = parser.parse_args(['backtest', 'ml-tracked', '--direction', 'short'])
        assert args.func == nat.cmd_backtest_ml_tracked
        assert args.direction == 'short'

    def test_backtest_list(self, parser):
        args = parser.parse_args(['backtest', 'list'])
        assert args.func == nat.cmd_backtest_list


# ── Model commands ───────────────────────────────────────────────────────────

class TestModel:
    def test_model_train(self, parser):
        args = parser.parse_args(['model', 'train', '--type', 'xgboost'])
        assert args.func == nat.cmd_model_train
        assert args.type == 'xgboost'

    def test_model_train_gmm(self, parser):
        args = parser.parse_args(['model', 'train-gmm', '--auto'])
        assert args.func == nat.cmd_model_train_gmm
        assert args.auto is True

    def test_model_list(self, parser):
        args = parser.parse_args(['model', 'list'])
        assert args.func == nat.cmd_model_list

    def test_model_score(self, parser):
        args = parser.parse_args(['model', 'score', '--save', '--model', 'my.pkl'])
        assert args.func == nat.cmd_model_score
        assert args.save is True
        assert args.model == 'my.pkl'

    def test_model_serve(self, parser):
        args = parser.parse_args(['model', 'serve', '--port', '9000', '--dev'])
        assert args.func == nat.cmd_model_serve
        assert args.port == 9000
        assert args.dev is True

    def test_model_serve_best(self, parser):
        args = parser.parse_args(['model', 'serve', '--best', '--metric', 'pnl'])
        assert args.func == nat.cmd_model_serve
        assert args.best is True
        assert args.metric == 'pnl'


# ── Cluster commands ─────────────────────────────────────────────────────────

class TestCluster:
    def test_cluster_analyze(self, parser):
        args = parser.parse_args(['cluster', 'analyze', '--symbol', 'SOL', '--hours', '12'])
        assert args.func == nat.cmd_cluster_analyze
        assert args.symbol == 'SOL'
        assert args.hours == 12

    def test_cluster_gmm(self, parser):
        args = parser.parse_args(['cluster', 'gmm'])
        assert args.func == nat.cmd_cluster_gmm

    def test_cluster_all(self, parser):
        args = parser.parse_args(['cluster', 'all'])
        assert args.func == nat.cmd_cluster_all


# ── Experiment tracking ──────────────────────────────────────────────────────

class TestExperiment:
    def test_experiment_list(self, parser):
        args = parser.parse_args(['experiment', 'list', '--stage', 'backtest'])
        assert args.func == nat.cmd_experiment_list
        assert args.stage == 'backtest'

    def test_experiment_get(self, parser):
        args = parser.parse_args(['experiment', 'get', '--id', 'exp_001'])
        assert args.func == nat.cmd_experiment_get
        assert args.id == 'exp_001'

    def test_experiment_compare(self, parser):
        args = parser.parse_args(['experiment', 'compare', '--ids', 'exp1 exp2'])
        assert args.func == nat.cmd_experiment_compare
        assert args.ids == 'exp1 exp2'

    def test_experiment_best(self, parser):
        args = parser.parse_args(['experiment', 'best', '--metric', 'pnl'])
        assert args.func == nat.cmd_experiment_best
        assert args.metric == 'pnl'

    def test_experiment_workflow(self, parser):
        args = parser.parse_args(['experiment', 'workflow', '--type', 'ridge'])
        assert args.func == nat.cmd_experiment_workflow
        assert args.type == 'ridge'


# ── Pipeline commands ────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_start(self, parser):
        args = parser.parse_args(['pipeline', 'start'])
        assert args.func == nat.cmd_pipeline_start

    def test_pipeline_resume(self, parser):
        args = parser.parse_args(['pipeline', 'resume'])
        assert args.func == nat.cmd_pipeline_resume

    def test_pipeline_analyze(self, parser):
        args = parser.parse_args(['pipeline', 'analyze'])
        assert args.func == nat.cmd_pipeline_analyze

    def test_pipeline_stop(self, parser):
        args = parser.parse_args(['pipeline', 'stop'])
        assert args.func == nat.cmd_pipeline_stop

    def test_pipeline_status(self, parser):
        args = parser.parse_args(['pipeline', 'status'])
        assert args.func == nat.cmd_pipeline_status

    def test_pipeline_dashboard(self, parser):
        args = parser.parse_args(['pipeline', 'dashboard', '--port', '9000'])
        assert args.func == nat.cmd_pipeline_dashboard
        assert args.port == 9000

    def test_pipeline_custom_config(self, parser):
        args = parser.parse_args(['pipeline', '--config', 'custom.toml', 'start'])
        assert args.func == nat.cmd_pipeline_start
        assert args.config == 'custom.toml'


# ── Signal commands ──────────────────────────────────────────────────────────

class TestSignal:
    def test_signal_test(self, parser):
        args = parser.parse_args(['signal', 'test', '--symbol', 'ETH', '--horizon', '6000'])
        assert args.func == nat.cmd_signal_test
        assert args.symbol == 'ETH'
        assert args.horizon == 6000

    def test_signal_test_remove_leaky(self, parser):
        args = parser.parse_args(['signal', 'test', '--remove-leaky'])
        assert args.func == nat.cmd_signal_test
        assert args.remove_leaky is True

    def test_signal_test_all(self, parser):
        args = parser.parse_args(['signal', 'test-all', '--spread-bps', '2.0'])
        assert args.func == nat.cmd_signal_test_all
        assert args.spread_bps == 2.0


# ── EAMM commands ────────────────────────────────────────────────────────────

class TestEAMM:
    def test_eamm_run(self, parser):
        args = parser.parse_args(['eamm', 'run', '--symbol', 'SOL', '--mode', 'classification'])
        assert args.func == nat.cmd_eamm_run
        assert args.symbol == 'SOL'
        assert args.mode == 'classification'

    def test_eamm_regime(self, parser):
        args = parser.parse_args(['eamm', 'regime'])
        assert args.func == nat.cmd_eamm_regime

    def test_eamm_backtest(self, parser):
        args = parser.parse_args(['eamm', 'backtest', '--gamma', '0.5', '--q-max', '2.0'])
        assert args.func == nat.cmd_eamm_backtest
        assert args.gamma == 0.5
        assert args.q_max == 2.0


# ── API commands ─────────────────────────────────────────────────────────────

class TestAPI:
    def test_api_start(self, parser):
        args = parser.parse_args(['api', 'start'])
        assert args.func == nat.cmd_api_start

    def test_api_alerts(self, parser):
        args = parser.parse_args(['api', 'alerts'])
        assert args.func == nat.cmd_api_alerts

    def test_api_serve_all(self, parser):
        args = parser.parse_args(['api', 'serve-all'])
        assert args.func == nat.cmd_api_serve_all


# ── Exp commands ─────────────────────────────────────────────────────────────

class TestExp:
    def test_exp_start(self, parser):
        args = parser.parse_args(['exp', 'start'])
        assert args.func == nat.cmd_exp_start

    def test_exp_stop(self, parser):
        args = parser.parse_args(['exp', 'stop'])
        assert args.func == nat.cmd_exp_stop

    def test_exp_status(self, parser):
        args = parser.parse_args(['exp', 'status'])
        assert args.func == nat.cmd_exp_status

    def test_exp_check(self, parser):
        args = parser.parse_args(['exp', 'check', '--hours', '48'])
        assert args.func == nat.cmd_exp_check
        assert args.hours == 48

    def test_exp_midweek(self, parser):
        args = parser.parse_args(['exp', 'midweek'])
        assert args.func == nat.cmd_exp_midweek

    def test_exp_analyze(self, parser):
        args = parser.parse_args(['exp', 'analyze'])
        assert args.func == nat.cmd_exp_analyze

    def test_exp_dashboard(self, parser):
        args = parser.parse_args(['exp', 'dashboard'])
        assert args.func == nat.cmd_exp_dashboard

    def test_exp_tunnel(self, parser):
        args = parser.parse_args(['exp', 'tunnel'])
        assert args.func == nat.cmd_exp_tunnel


# ── Docker commands ──────────────────────────────────────────────────────────

class TestDocker:
    def test_docker_build(self, parser):
        args = parser.parse_args(['docker', 'build'])
        assert args.func == nat.cmd_docker_build

    def test_docker_up(self, parser):
        args = parser.parse_args(['docker', 'up'])
        assert args.func == nat.cmd_docker_up

    def test_docker_down(self, parser):
        args = parser.parse_args(['docker', 'down'])
        assert args.func == nat.cmd_docker_down

    def test_docker_logs(self, parser):
        args = parser.parse_args(['docker', 'logs'])
        assert args.func == nat.cmd_docker_logs

    def test_docker_ps(self, parser):
        args = parser.parse_args(['docker', 'ps'])
        assert args.func == nat.cmd_docker_ps

    def test_docker_stack(self, parser):
        args = parser.parse_args(['docker', 'stack'])
        assert args.func == nat.cmd_docker_stack

    def test_docker_stack_no_build(self, parser):
        args = parser.parse_args(['docker', 'stack', '--no-build'])
        assert args.func == nat.cmd_docker_stack
        assert args.no_build is True

    def test_docker_smoke(self, parser):
        args = parser.parse_args(['docker', 'smoke'])
        assert args.func == nat.cmd_docker_smoke


# ── Swarm commands ──────────────────────────────────────────────────────────

class TestSwarm:
    def test_swarm_run_defaults(self, parser):
        args = parser.parse_args(['swarm', 'run'])
        assert args.func == nat.cmd_swarm_run
        assert args.instances == 16
        assert args.hours == 24
        assert args.symbol == 'BTC'
        assert args.seed is None
        assert args.workers is None
        assert args.json is False

    def test_swarm_run_custom(self, parser):
        args = parser.parse_args(['swarm', 'run', '--instances', '8',
                                   '--hours', '4', '--symbol', 'ETH',
                                   '--seed', '42', '--workers', '4'])
        assert args.func == nat.cmd_swarm_run
        assert args.instances == 8
        assert args.hours == 4
        assert args.symbol == 'ETH'
        assert args.seed == 42
        assert args.workers == 4

    def test_swarm_run_short_flag(self, parser):
        args = parser.parse_args(['swarm', 'run', '-n', '4'])
        assert args.instances == 4

    def test_swarm_status(self, parser):
        args = parser.parse_args(['swarm', 'status'])
        assert args.func == nat.cmd_swarm_status

    def test_swarm_results_defaults(self, parser):
        args = parser.parse_args(['swarm', 'results'])
        assert args.func == nat.cmd_swarm_results
        assert args.top == 10
        assert args.json is False

    def test_swarm_results_custom(self, parser):
        args = parser.parse_args(['swarm', 'results', '--top', '5', '--json'])
        assert args.func == nat.cmd_swarm_results
        assert args.top == 5
        assert args.json is True

    def test_swarm_best_defaults(self, parser):
        args = parser.parse_args(['swarm', 'best'])
        assert args.func == nat.cmd_swarm_best
        assert args.export == 'config/best_algorithms.toml'

    def test_swarm_best_custom(self, parser):
        args = parser.parse_args(['swarm', 'best', '--export', '/tmp/out.toml'])
        assert args.func == nat.cmd_swarm_best
        assert args.export == '/tmp/out.toml'

    def test_swarm_generate_defaults(self, parser):
        args = parser.parse_args(['swarm', 'generate'])
        assert args.func == nat.cmd_swarm_generate
        assert args.count == 16
        assert args.output == 'data/swarm/configs'
        assert args.seed is None

    def test_swarm_generate_custom(self, parser):
        args = parser.parse_args(['swarm', 'generate', '-n', '32',
                                   '--seed', '123', '--output', '/tmp/configs'])
        assert args.func == nat.cmd_swarm_generate
        assert args.count == 32
        assert args.seed == 123
        assert args.output == '/tmp/configs'


class TestEvolve:
    def test_evolve_start_defaults(self, parser):
        args = parser.parse_args(['evolve', 'start'])
        assert args.func == nat.cmd_evolve_start
        assert args.study == 'nat_evolve'
        assert args.sampler == 'cma'
        assert args.trials == 500
        assert args.jobs == 1
        assert args.symbol == 'BTC'
        assert args.hours == 720
        assert abs(args.train_frac - 0.667) < 0.01
        assert args.seed == 42
        assert args.timeout is None
        assert args.no_guard_rails is False

    def test_evolve_start_custom(self, parser):
        args = parser.parse_args(['evolve', 'start', '--study', 'test',
                                   '--sampler', 'nsga2', '--trials', '100',
                                   '--jobs', '4', '--symbol', 'ETH',
                                   '--hours', '168', '--seed', '7',
                                   '--timeout', '3600', '--no-guard-rails'])
        assert args.func == nat.cmd_evolve_start
        assert args.study == 'test'
        assert args.sampler == 'nsga2'
        assert args.trials == 100
        assert args.jobs == 4
        assert args.symbol == 'ETH'
        assert args.hours == 168
        assert args.seed == 7
        assert args.timeout == 3600
        assert args.no_guard_rails is True

    def test_evolve_start_short_flag(self, parser):
        args = parser.parse_args(['evolve', 'start', '-n', '50'])
        assert args.trials == 50

    def test_evolve_status(self, parser):
        args = parser.parse_args(['evolve', 'status'])
        assert args.func == nat.cmd_evolve_status
        assert args.study == 'nat_evolve'

    def test_evolve_best_defaults(self, parser):
        args = parser.parse_args(['evolve', 'best'])
        assert args.func == nat.cmd_evolve_best
        assert args.top == 5
        assert args.json is False

    def test_evolve_best_custom(self, parser):
        args = parser.parse_args(['evolve', 'best', '--top', '10',
                                   '--json', '--study', 'my_study'])
        assert args.func == nat.cmd_evolve_best
        assert args.top == 10
        assert args.json is True
        assert args.study == 'my_study'

    def test_evolve_pareto(self, parser):
        args = parser.parse_args(['evolve', 'pareto'])
        assert args.func == nat.cmd_evolve_pareto
        assert args.study == 'nat_evolve'

    def test_evolve_export_defaults(self, parser):
        args = parser.parse_args(['evolve', 'export'])
        assert args.func == nat.cmd_evolve_export
        assert args.output == 'config/evolved_algorithms.toml'

    def test_evolve_export_custom(self, parser):
        args = parser.parse_args(['evolve', 'export', '--output', '/tmp/best.toml',
                                   '--study', 'prod_study'])
        assert args.func == nat.cmd_evolve_export
        assert args.output == '/tmp/best.toml'
        assert args.study == 'prod_study'


# ── CLI smoke tests (subprocess) ─────────────────────────────────────────────

class TestCLISmoke:
    def test_help_output(self):
        r = subprocess.run([sys.executable, str(ROOT / 'nat'), 'help'],
                          capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert 'nat' in r.stdout
        assert 'build' in r.stdout.lower()
        assert 'backtest' in r.stdout.lower()
        assert 'pipeline' in r.stdout.lower()

    def test_no_args_shows_help(self):
        r = subprocess.run([sys.executable, str(ROOT / 'nat')],
                          capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert 'nat' in r.stdout

    def test_all_subcommand_help(self):
        for cmd in ['build', 'run', 'data', 'test', 'backtest', 'model',
                    'cluster', 'experiment', 'pipeline', 'signal', 'eamm',
                    'api', 'exp', 'docker', 'swarm', 'evolve']:
            r = subprocess.run([sys.executable, str(ROOT / 'nat'), cmd, '--help'],
                              capture_output=True, text=True, timeout=10)
            assert r.returncode == 0, f'{cmd} --help failed: {r.stderr}'

    def test_status_runs(self):
        r = subprocess.run([sys.executable, str(ROOT / 'nat'), 'status'],
                          capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert 'Ingestor' in r.stdout
