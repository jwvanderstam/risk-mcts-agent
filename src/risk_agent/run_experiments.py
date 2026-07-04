import argparse
import copy
import logging
import logging.config
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from risk_agent.engine.battle_computer import BattleComputer
from risk_agent.engine.game_manager import GameManager
from risk_agent.utils.data_collector import DataCollector

logger = logging.getLogger(__name__)


def get_allocated_cpus() -> int:
    """
    Gets the number of CPUs allocated by Slurm,
    defaulting to the machine's total count if not on Slurm.
    """
    try:
        # Read the Slurm environment variable
        n_cpus = (
            int(os.environ['SLURM_CPUS_PER_TASK']) - 1
        )  # Reserve one CPU for the main process
        logger.info(f'Running on Slurm. Allocated CPUs: {n_cpus}')
    except KeyError:
        # Variable not set, so not running on Slurm
        logger.info('Not running on Slurm, using os.cpu_count().')
        n_cpus = os.cpu_count()
        if n_cpus is not None:
            n_cpus = max(1, n_cpus - 1)  # Reserve one CPU for the main process
        else:
            n_cpus = 1  # Fallback to 1 if cpu_count() returns None

    return n_cpus


def set_nested_value(d: dict, key_path: str, value: Any) -> None:  # noqa: ANN401
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def initialize_battle_computer(exp_settings: dict) -> None:
    """
    Pre-initialize the BattleComputer to ensure the stationary distribution
    file exists before starting multiprocessing. This avoids race conditions
    where multiple processes try to create the file simultaneously.

    Args:
        exp_settings: Experiment settings containing game configuration
    """
    logger.info('Pre-initializing BattleComputer...')
    start_time = time.time()

    _ = BattleComputer(
        max_attacking_armies=exp_settings['game_settings']['max_attacking_armies'],
        max_defending_armies=exp_settings['game_settings']['max_defending_armies'],
    )

    end_time = time.time()
    logger.info(f'BattleComputer initialized in {end_time - start_time:.2f} seconds.')


def run_single_game(args_tuple: tuple) -> dict:
    """
    Run a single game instance. This function is called by each worker process.

    Args:
        args_tuple: Tuple containing (run_id, exp_settings, logging_config, seed, experiment_output_dir)

    Returns:
        Dictionary containing game-level data (action, turn, and player data saved to disk)
    """
    run_id, exp_settings, logging_config, seed, experiment_output_dir = args_tuple

    # Set up logging for this process
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    random.seed(seed)

    logger.info(f'Starting run {run_id} in process {os.getpid()}')
    run_start_time = time.time()

    try:
        # Create battle computer for this process
        # This will load the pre-computed stationary distribution from disk
        # Each process gets its own instance with independent random state
        battle_computer = BattleComputer(
            max_attacking_armies=exp_settings['game_settings']['max_attacking_armies'],
            max_defending_armies=exp_settings['game_settings']['max_defending_armies'],
        )

        # Create data collector for this process
        # Use the actual experiment output directory
        data_collector = DataCollector(
            experiment_path=experiment_output_dir,
            number_of_players=exp_settings['game_settings']['number_of_players'],
            player_types=[player['type'] for player in exp_settings['players']],
            max_turns=exp_settings['game_settings']['max_turns'],
            max_attacking_armies=exp_settings['game_settings']['max_attacking_armies'],
            max_defending_armies=exp_settings['game_settings']['max_defending_armies'],
        )
        data_collector.game_id = run_id

        # Initialize GameManager
        game_manager = GameManager(logging_config=logging_config)
        game_manager.settings = copy.deepcopy(exp_settings)
        game_manager.process_settings()
        game_manager.data_collector = data_collector
        game_manager.battle_computer = battle_computer

        # Randomize player positions if specified
        if exp_settings['randomize_player_positions']:
            random.shuffle(game_manager.settings['players'])
            logger.info(
                f'Run {run_id}: Randomized player types: {game_manager.settings["players"]}'
            )
            data_collector.player_types = [
                player['type'] for player in game_manager.settings['players']
            ]

        # Run the game
        game_manager.run_game()

        run_end_time = time.time()
        logger.info(
            f'Run {run_id} finished in {run_end_time - run_start_time:.2f} seconds.'
        )

        # Save action, turn, and player level data immediately (per-run files)
        save_run_level_data(data_collector, experiment_output_dir)

        # Return only game-level data to be aggregated
        return {
            'run_id': run_id,
            'experiment_name': exp_settings['experiment_name'],
            'game_data': data_collector.game_data,
            'success': True,
            'error': None,
        }

    except Exception:
        logger.exception(f'Error during run {run_id}')
        return {
            'run_id': run_id,
            'experiment_name': exp_settings['experiment_name'],
            'game_data': [],
            'success': False,
            'error': str(e),
        }


def save_run_level_data(
    data_collector: DataCollector, experiment_output_dir: str
) -> None:
    """
    Save action-level, turn-level, and player-level data for a single run.
    These are saved as individual files per run.

    Args:
        data_collector: DataCollector instance with collected data
        experiment_output_dir: Base directory for the experiment
    """
    logger = logging.getLogger(__name__)
    run_id = data_collector.game_id
    number_of_players = data_collector.number_of_players

    try:
        # Save action-level data for this run
        if data_collector.action_data:
            pd.DataFrame(data_collector.action_data).to_csv(
                f'{experiment_output_dir}/action_level/run_{run_id}.csv', index=False
            )

        # Save turn-level data for this run
        if data_collector.turn_data:
            pd.DataFrame(data_collector.turn_data).to_csv(
                f'{experiment_output_dir}/turn_level/run_{run_id}.csv', index=False
            )

        # Save player-level data for this run
        for player_id in range(number_of_players):
            if data_collector.player_data[player_id]:
                pd.DataFrame(data_collector.player_data[player_id]).to_csv(
                    f'{experiment_output_dir}/player_level/run_{run_id}_player_{player_id}_data.csv',
                    index=False,
                )

        logger.info(f'Run {run_id}: Saved action, turn, and player level data')

    except Exception:
        logger.exception(f'Error saving run-level data for run {run_id}')


def aggregate_and_save_data(results: list, experiment_output_dir: str) -> None:
    """
    Aggregate game-level data from all runs and save to a single CSV file.
    Action, turn, and player level data are already saved as individual files per run.

    Args:
        results: List of result dictionaries from run_single_game
        experiment_output_dir: Directory to save the aggregated data
    """
    logger.info(f'Aggregating game-level data for {experiment_output_dir}...')

    # Ensure output directory exists
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Aggregate only game-level data
    all_game_data = []

    successful_runs = 0
    failed_runs = 0

    for result in results:
        if result['success']:
            successful_runs += 1
            all_game_data.extend(result['game_data'])
        else:
            failed_runs += 1
            logger.warning(
                f'Run {result["run_id"]} failed with error: {result["error"]}'
            )

    logger.info(f'Successful runs: {successful_runs}, Failed runs: {failed_runs}')

    # Save aggregated game-level data
    try:
        if all_game_data:
            pd.DataFrame(all_game_data).to_csv(
                f'{experiment_output_dir}/game_level.csv', index=False
            )
            logger.info(
                f'Game-level data successfully saved to {experiment_output_dir}/game_level.csv'
            )
        else:
            logger.warning('No game-level data to save')

    except Exception:
        logger.exception('Error saving game-level data')


def prepare_experiment_runs(exp_settings: dict, experiment_output_dir: str) -> tuple:
    """
    Prepare run arguments for an experiment, checking for existing runs.

    Args:
        exp_settings: Experiment configuration dictionary
        experiment_output_dir: Directory for experiment output

    Returns:
        Tuple of (run_args, previously_completed_runs, existing_game_data)
    """
    exp_name = exp_settings['experiment_name']
    num_runs = exp_settings['runs']

    # Check for existing data to resume
    previously_completed_runs = 0
    existing_game_data = None
    if os.path.exists(f'{experiment_output_dir}/game_level.csv'):
        existing_game_data = pd.read_csv(f'{experiment_output_dir}/game_level.csv')
        previously_completed_runs = len(existing_game_data)
        logger.info(
            f'{exp_name}: Found {previously_completed_runs} previously completed runs. Resuming...'
        )

    # Prepare arguments for each run
    run_args = []
    for i in range(num_runs):
        if i < previously_completed_runs:
            logger.info(f'{exp_name}: Skipping already completed run {i + 1}')
            continue

        # Generate a unique seed for each run for reproducibility
        seed = hash(f'{exp_name}_{i}') % (2**32)
        run_args.append((i + 1, exp_settings, None, seed, experiment_output_dir))

    return run_args, previously_completed_runs, existing_game_data


def finalize_experiment_results(
    exp_name: str,
    results: list,
    experiment_output_dir: str,
    number_of_players: int,
    previously_completed_runs: int,
    existing_game_data: pd.DataFrame = None,
) -> None:
    """
    Finalize and save results for a single experiment.

    Args:
        exp_name: Experiment name
        results: List of result dictionaries for this experiment
        experiment_output_dir: Directory for experiment output
        number_of_players: Number of players in the game
        previously_completed_runs: Number of runs completed before this batch
        existing_game_data: Previously saved game data (if resuming)
    """
    # Filter results for this experiment only
    exp_results = [r for r in results if r['experiment_name'] == exp_name]

    if not exp_results:
        logger.warning(f'{exp_name}: No results to save')
        return

    # If we're resuming, combine existing data with new results
    if existing_game_data is not None:
        logger.info(
            f'{exp_name}: Combining existing game-level data with new results...'
        )
        # Create dummy results for skipped runs to maintain proper structure
        existing_results = []
        for i in range(previously_completed_runs):
            existing_results.append(
                {
                    'run_id': i + 1,
                    'experiment_name': exp_name,
                    'game_data': existing_game_data.iloc[i : i + 1].to_dict('records'),
                    'success': True,
                    'error': None,
                }
            )
        exp_results = existing_results + exp_results

    # Aggregate and save game-level data
    aggregate_and_save_data(exp_results, experiment_output_dir)


def run_experiment(
    exp_settings: dict, logging_config: dict, num_processes: int
) -> None:
    """
    Runs a single experiment based on the provided configuration using multiprocessing.

    Args:
        exp_settings: Experiment configuration dictionary
        logging_config: Logging configuration dictionary
        num_processes: Number of processes to use (defaults to CPU count - 1)
    """
    exp_name = exp_settings['experiment_name']
    num_runs = exp_settings['runs']

    logger.info(
        f'Starting Experiment: {exp_name} for {num_runs} runs using {num_processes} processes'
    )
    experiment_start_time = time.time()

    # Pre-initialize BattleComputer to create stationary distribution file
    # This avoids race conditions when multiple processes start
    initialize_battle_computer(exp_settings)

    experiment_output_dir = (
        f'{exp_settings["paths"]["experiment_output_path"]}/{exp_name}'
    )

    # Ensure output directories exist before multiprocessing
    os.makedirs(experiment_output_dir, exist_ok=True)
    os.makedirs(f'{experiment_output_dir}/action_level', exist_ok=True)
    os.makedirs(f'{experiment_output_dir}/turn_level', exist_ok=True)
    os.makedirs(f'{experiment_output_dir}/player_level', exist_ok=True)

    # Prepare run arguments
    run_args, previously_completed_runs, existing_game_data = prepare_experiment_runs(
        exp_settings, experiment_output_dir
    )

    if not run_args:
        logger.info(f'{exp_name}: All runs already completed. Nothing to do.')
        return

    # Add logging config to run args
    run_args = [(r[0], r[1], logging_config, r[3], r[4]) for r in run_args]

    # Run games in parallel
    logger.info(f'{exp_name}: Running {len(run_args)} games in parallel...')

    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_game, run_args)

    # Finalize results
    finalize_experiment_results(
        exp_name,
        results,
        experiment_output_dir,
        exp_settings['game_settings']['number_of_players'],
        previously_completed_runs,
        existing_game_data,
    )

    experiment_end_time = time.time()
    logger.info(
        f'Experiment {exp_name} completed in {experiment_end_time - experiment_start_time:.2f} seconds.'
    )


def run_tuning_experiment(
    base_config: dict, logging_config: dict, num_processes: int
) -> None:
    """
    Runs a parameter tuning experiment with multiprocessing across all parameter values.
    All runs from all parameter values are submitted to a single pool, eliminating
    the bottleneck where CPUs wait for the slowest run of each parameter value.
    """
    param_path = base_config['tuning_parameter']
    param_values = base_config['tuning_values']

    logger.info(
        f'Starting tuning experiment: {base_config["experiment_name"]} '
        f'with {len(param_values)} parameter values'
    )
    tuning_start_time = time.time()

    # Prepare all experiment configurations
    exp_configs = []
    exp_metadata = {}  # Track metadata for each experiment

    for value in param_values:
        exp_config = copy.deepcopy(base_config)
        set_nested_value(exp_config, param_path, value)
        exp_name = f'{base_config["experiment_name"]}_{value}'
        exp_config['experiment_name'] = exp_name

        # Pre-initialize BattleComputer for this configuration
        initialize_battle_computer(exp_config)

        experiment_output_dir = (
            f'{exp_config["paths"]["experiment_output_path"]}/{exp_name}'
        )

        # Ensure output directories exist
        os.makedirs(experiment_output_dir, exist_ok=True)
        os.makedirs(f'{experiment_output_dir}/action_level', exist_ok=True)
        os.makedirs(f'{experiment_output_dir}/turn_level', exist_ok=True)
        os.makedirs(f'{experiment_output_dir}/player_level', exist_ok=True)

        exp_configs.append((exp_config, experiment_output_dir))

    # Prepare all run arguments from all experiments
    all_run_args = []

    for exp_config, experiment_output_dir in exp_configs:
        run_args, previously_completed, existing_data = prepare_experiment_runs(
            exp_config, experiment_output_dir
        )

        # Store metadata for later aggregation
        exp_metadata[exp_config['experiment_name']] = {
            'output_dir': experiment_output_dir,
            'num_players': exp_config['game_settings']['number_of_players'],
            'previously_completed': previously_completed,
            'existing_data': existing_data,
        }

        # Add logging config to run args
        run_args = [(r[0], r[1], logging_config, r[3], r[4]) for r in run_args]
        all_run_args.extend(run_args)

    if not all_run_args:
        logger.info('All runs for all parameter values already completed.')
        return

    logger.info(
        f'Running {len(all_run_args)} total games across {len(param_values)} '
        f'parameter values using {num_processes} processes'
    )

    # Run ALL games from ALL parameter values in a single pool
    with Pool(processes=num_processes) as pool:
        all_results = pool.map(run_single_game, all_run_args)

    # Finalize results for each experiment
    for exp_name, metadata in exp_metadata.items():
        finalize_experiment_results(
            exp_name,
            all_results,
            metadata['output_dir'],
            metadata['num_players'],
            metadata['previously_completed'],
            metadata['existing_data'],
        )

    tuning_end_time = time.time()
    logger.info(
        f'Tuning experiment {base_config["experiment_name"]} completed in '
        f'{tuning_end_time - tuning_start_time:.2f} seconds.'
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run Risk game experiments with multiprocessing.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/test_new.yaml',
        help='Path to the experiments configuration file.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to load configurations and run all experiments.
    """
    args = parse_args()

    # Load logging configuration
    with open('./data/logging.yaml') as file:
        logging_config = yaml.safe_load(file)
    logging.config.dictConfig(logging_config)

    try:
        # Resolve and confine the config path to the current working
        # directory: --config is untrusted CLI input, and without this check
        # a value like '../../../etc/passwd' would let the caller read any
        # file the process has access to.
        cwd = Path.cwd().resolve()
        config_path = Path(args.config).resolve()
        if cwd != config_path and cwd not in config_path.parents:
            logger.error(f'Config path {args.config} must be within {cwd}.')
            return

        with open(config_path) as f:
            experiments_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f'File {args.config} not found.')
        return
    except yaml.YAMLError:
        logger.exception(f'Error parsing {args.config}')
        return

    # Determine number of processes based on Slurm allocation
    allocated_cpus = get_allocated_cpus()

    for base_config in experiments_config['experiments']:
        exp_type = base_config['experiment_type']

        if exp_type == 'standard':
            run_experiment(base_config, logging_config, allocated_cpus)
        elif exp_type == 'tuning':
            run_tuning_experiment(base_config, logging_config, allocated_cpus)


if __name__ == '__main__':
    main()
