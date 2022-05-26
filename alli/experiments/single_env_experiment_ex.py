from collections import defaultdict
import sys
import socket
from timeit import default_timer as timer
from typing import OrderedDict

import numpy as np
from scipy import stats
import torch

from all.experiments.writer import CometWriter
from all.experiments.experiment import Experiment
from alli.experiments.writer_ex import ExperimentWriterEx
from rlkit.core.logging import logger_main, logger_sub

if '--debug_profile' in sys.argv:
    from rlkit.core.logging import profile
else:
    profile = lambda func: func

g_curr_hostname = socket.gethostname()

class SingleEnvExperimentEx(Experiment):
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''

    def __init__(
            self,
            preset,
            env,
            name=None,
            train_steps=float('inf'),
            num_test_episodes=100,
            test_frequency_in_episodes=None,
            test_light_frequency_in_episodes=None,
            save_frequency_in_episodes=None,
            logdir='runs',
            dir_name=None,
            run_group=None,
            quiet=False,
            render=False,
            write_loss=True,
            writer="tensorboard",
            write_to_tensorboard_events=False,
            on_dirs_created=None,
    ):
        self._name = name if name is not None else preset.name
        super().__init__(self._make_writer(logdir, self._name, env.name, write_loss, writer, dir_name, run_group, write_to_tensorboard_events, on_dirs_created), quiet)
        self._logdir = logdir
        self._preset = preset
        self._agent = self._preset.agent(writer=self._writer, train_steps=train_steps)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1
        self._num_test_episodes = num_test_episodes
        self._test_frequency_in_episodes = test_frequency_in_episodes
        self._test_light_frequency_in_episodes = test_light_frequency_in_episodes
        self._save_frequency_in_episodes = save_frequency_in_episodes
        self._last_test_ex_result = None
        self._best_returns_discounted = -np.inf
        self._returns_discounted100 = []

        if render:
            self._env.render(mode="human")

    @property
    def agent(self):
        return self._agent

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf, debug_no_initial_test=False):
        self._train_start_time = timer()
        self._train_start_frame = self._frame
        while not self._done(frames, episodes):
            self._run_training_episode(debug_no_initial_test=debug_no_initial_test)

            if '--debug_profile' in sys.argv and self._episode % 11 == 0:
                from rlkit.core.logging import profiler
                profiler.print_stats()
                aa = input('Paused.....................')

    def test(self, episodes=100):
        test_agent = self._preset.test_agent()
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode(test_agent)
            returns.append(episode_return)
            self._log_test_episode(episode, episode_return)
        self._log_test(returns)
        return returns

    def _process_test_agents_dict(self, episodes, test_agents_dict, results, dummy=False):
        for name, test_agent in test_agents_dict.items():
            if dummy and name in results:
                continue
            results[name] = defaultdict(list)
            results[name]['_is_dummy'] = dummy
            for episode in range(episodes):
                if dummy:
                    results[name]['returns'].append(0.0)
                    results[name]['returns_discounted'].append(0.0)
                    #results[name]['returns_normalized'].append(0.0)
                else:
                    episode_results = self._run_test_episode(test_agent)
                    results[name]['returns'].append(episode_results['return'])
                    results[name]['returns_discounted'].append(episode_results['return_discounted'])
                    #results[name]['returns_normalized'].append(episode_results['return_normalized'])
                    self._log_test_ex_episode(
                        name=name,
                        episode=episode,
                        results=episode_results)
            if name.startswith('GPI:') or name.startswith('GPISourceTaskVecsAnd:'):
                gk = ('GPISingleVec' if name.startswith('GPI:') else 'GPISingleVecAndSourceTasks')
                if gk not in results:
                    results[gk] = defaultdict(list)
                results[gk]['mean_returns'].append(np.mean(results[name]['returns']))
                results[gk]['std_returns'].append(np.std(results[name]['returns']))
                results[gk]['mean_returns_discounted'].append(np.mean(results[name]['returns_discounted']))
                results[gk]['std_returns_discounted'].append(np.std(results[name]['returns_discounted']))
                #results[gk]['mean_returns_normalized'].append(np.mean(results[name]['returns_normalized']))
                #results[gk]['std_returns_normalized'].append(np.std(results[name]['returns_normalized']))

    def test_ex(self, episodes=100, light_test=False):
        if self._last_test_ex_result is None:
            results = OrderedDict()
        else:
            results = self._last_test_ex_result

        test_agents_info = self._preset.test_agents(
            determine_dummy=lambda x: (light_test and x == 'other_test_agents'))
        self._process_test_agents_dict(episodes, test_agents_info['light_test_agents'], results, dummy=False)
        self._process_test_agents_dict(episodes, test_agents_info['other_test_agents'], results, dummy=light_test)

        for name, r in results.items():
            self._log_test_ex(name=name, results=r)

        self._last_test_ex_result = results
        return results

    def test_ex_dummy(self, episodes=100):
        if self._last_test_ex_result is None:
            results = OrderedDict()
        else:
            results = self._last_test_ex_result

        test_agents_info = self._preset.test_agents(
            determine_dummy=lambda x: True)
        self._process_test_agents_dict(episodes, test_agents_info['light_test_agents'], results, dummy=True)
        self._process_test_agents_dict(episodes, test_agents_info['other_test_agents'], results, dummy=True)

        self._last_test_ex_result = results

        for name, r in self._last_test_ex_result.items():
            if not (name.startswith('GPI:') or name.startswith('GPISourceTaskVecsAnd:')):
                self._log_test_ex(name=name, results=r)

        return self._last_test_ex_result

    def _log_test_ex_episode(self, name, episode, results):
        if not self._quiet:
            print('test ({}) episode: {}, {}'.format(name, episode, ', '.join(f'{k}: {v}' for k, v in results.items())))

    def _log_test_ex(self, name, results):
        is_dummy = results.pop('_is_dummy', False)
        if not self._quiet:
            print('test ({}) {}'.format(
                name,
                ', '.join(f'{k} (mean ± sem): {np.mean(v)} ± {stats.sem(v)}' for k, v in results.items())))
        for k, v in results.items():
            if name.startswith('GPI:') or name.startswith('GPISourceTaskVecsAnd:'):
                logger_sub.record_tabular(f'{name}__{k}__test/mean', np.mean(v))
                logger_sub.record_tabular(f'{name}__{k}__test/std', np.std(v))
                logger_sub.record_tabular(f'{name}__{k}__test/max', np.max(v))
                logger_sub.record_tabular(f'{name}__{k}__test/min', np.min(v))
            else:
                self._writer.add_summary(
                    f'{name}__{k}__test', np.mean(v), np.std(v), np.max(v), np.min(v),
                    write_to_tensorboard_events=(not is_dummy),
                )

    @profile
    def _run_training_episode(self, debug_no_initial_test):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        if not debug_no_initial_test and self._test_frequency_in_episodes is not None and self._episode == 1:
            with torch.no_grad():
                #results = self.test_ex(self._num_test_episodes)
                results = self.test_ex(3, light_test=False)
                self._preset.on_test_ex(results, self._writer, self.frame - 1, self.episode, self.agent)

            logger_sub.record_tabular('frame', self.frame)
            logger_sub.record_tabular('episode', self.episode)
            logger_sub.dump_tabular(with_prefix=False, with_timestamp=False)

        # initialize the episode
        state = self._env.reset()
        action = self._agent.act(state)
        returns = 0

        discount = 1.0
        returns_discounted = 0

        # loop until the episode is finished
        while not state.done:
            if self._frame != start_frame:
                self._writer.flush_csv_per_frame(flush_file=False)
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = self._agent.act(state)
            returns += state.reward
            returns_discounted += state.reward * discount
            discount *= self._preset.discount_factor
            self._writer.add_scalar('frame', self.frame, dump_targets=['frame'])
            self._writer.add_scalar('episode', self.episode, dump_targets=['frame'])
            self._frame += 1

        if self._frame != start_frame:
            self._writer.flush_csv_per_frame(flush_file=(self._episode % 100 == 0))

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)
        logger_main.log(f'[{g_curr_hostname}] [{self._episode}] fps: {fps}')

        # log the results
        self._log_training_episode(returns, returns_discounted, fps)

        need_test = (
            self._test_frequency_in_episodes is not None and
            self._episode % self._test_frequency_in_episodes == 0)
        need_light_test = (
            self._test_light_frequency_in_episodes is not None and
            self._episode % self._test_light_frequency_in_episodes == 0)

        if need_test or need_light_test:
            with torch.no_grad():
                results = self.test_ex(self._num_test_episodes, light_test=(not need_test))
                self._preset.on_test_ex(results, self._writer, self.frame - 1, self.episode, self.agent)

            logger_sub.record_tabular('frame', self.frame)
            logger_sub.record_tabular('episode', self.episode)
            logger_sub.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            _ = self.test_ex_dummy(self._num_test_episodes)

        fps_total = (self._frame - self._train_start_frame) / (timer() - self._train_start_time)
        logger_main.log(f'[{g_curr_hostname}] [{self._episode}] fps_total: {fps_total}')
        self._writer.add_scalar('fps_total', fps_total, dump_targets=['episode'])

        self._writer.add_scalar('frame', self.frame, dump_targets=['episode'])
        self._writer.add_scalar('episode', self.episode, dump_targets=['episode'])
        self._writer.flush_csv(skip_text_logging=True)

        if self._save_frequency_in_episodes is not None and self._episode % self._save_frequency_in_episodes == 0:
            self.save()

        # update experiment state
        self._episode += 1

    def _log_training_episode(self, returns, returns_discounted, fps):
        if not self._quiet:
            print('episode: {}, frame: {}, fps: {}, returns: {}, returns_discounted: {}'.format(self.episode, self.frame, int(fps), returns, returns_discounted))
        if returns > self._best_returns:
            self._best_returns = returns
        if returns_discounted > self._best_returns_discounted:
            self._best_returns_discounted = returns_discounted

        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            max = np.max(self._returns100)
            min = np.min(self._returns100)
            self._writer.add_summary('returns100', mean, std, max, min, step="frame")
            self._returns100 = []
        elif self._episode == 1:
            # This fixes the CSV key set mismatch.
            self._writer.add_summary('returns100', 0.0, 0.0, 0.0, 0.0, step="frame")
        self._returns_discounted100.append(returns)
        if len(self._returns_discounted100) == 100:
            mean = np.mean(self._returns_discounted100)
            std = np.std(self._returns_discounted100)
            max = np.max(self._returns_discounted100)
            min = np.min(self._returns_discounted100)
            self._writer.add_summary('returns_discounted100', mean, std, max, min, step="frame")
            self._returns_discounted100 = []
        elif self._episode == 1:
            # This fixes the CSV key set mismatch.
            self._writer.add_summary('returns_discounted100', 0.0, 0.0, 0.0, 0.0, step="frame")

        self._writer.add_evaluation('returns/episode', returns, step="episode", write_to_tensorboard_events=True)
        self._writer.add_evaluation('returns/frame', returns, step="frame")
        self._writer.add_evaluation("returns/max", self._best_returns, step="frame", write_to_tensorboard_events=True)
        self._writer.add_evaluation('returns_discounted/episode', returns_discounted, step="episode")
        self._writer.add_evaluation('returns_discounted/frame', returns_discounted, step="frame")
        self._writer.add_evaluation("returns_discounted/max", self._best_returns_discounted, step="frame")
        self._writer.add_scalar('fps', fps, step="frame")

    def save(self):
        return self._preset.save(f'{self._writer.log_dir}/preset_{self._episode:09d}.pt')

    def _run_test_episode(self, test_agent):
        # initialize the episode
        state = self._env.reset()
        action = test_agent.act(state)
        results = defaultdict(float)

        discount = 1.0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = test_agent.act(state)
            results['return'] += state.reward
            results['return_discounted'] += state.reward * discount
            discount *= self._preset.discount_factor
            if 'reward_normalized' in state.keys():
                results['return_normalized'] += state['reward_normalized']

        return results

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer, dir_name, run_group, write_to_tensorboard_events, on_dirs_created):
        if writer == "comet":
            return CometWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
        return ExperimentWriterEx(self, agent_name, env_name, loss=write_loss, logdir=logdir, dir_name=dir_name, run_group=run_group, write_to_tensorboard_events=write_to_tensorboard_events, on_dirs_created=on_dirs_created)

