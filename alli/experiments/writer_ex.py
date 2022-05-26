
import os
import csv
import subprocess
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from all.logging import Writer
from alli.core.io_util import GzipFileWrapper, LZ4FileWrapper
from rlkit.core.logging import logger_main, logger_sub, logger_per_frame


class ExperimentWriterEx(SummaryWriter, Writer):
    '''
    The default Writer object used by all.experiments.Experiment.
    Writes logs using tensorboard into the current logdir directory ('runs' by default),
    tagging the run with a combination of the agent name, the commit hash of the
    current git repo of the working directory (if any), and the current time.
    Also writes summary statistics into CSV files.
    Args:
        experiment (all.experiments.Experiment): The Experiment associated with the Writer object.
        agent_name (str): The name of the Agent the Experiment is being performed on
        loss (bool, optional): Whether or not to log loss/scheduling metrics, or only evaluation and summary metrics.
    '''

    def __init__(self,
                 experiment,
                 agent_name,
                 env_name,
                 loss=True,
                 grad_norm=True,
                 logdir='runs',
                 dir_name=None,
                 run_groups_dir='groups',
                 run_group=None,
                 write_to_tensorboard_events=False,
                 on_dirs_created=None,
                 ):
        if dir_name is None:
            current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S_%f')
            dir_name = "%s_%s_%s" % (agent_name, COMMIT_HASH, current_time)
        group_dir = None
        if run_group is not None:
            group_dir = os.path.join(run_groups_dir, run_group)
            try:
                os.makedirs(group_dir)
            except OSError:
                pass

            try:
                os.symlink(os.path.realpath(os.path.join(logdir, dir_name)), os.path.join(group_dir, dir_name))
            except (OSError, FileExistsError):
                pass
        self.log_dir = os.path.join(logdir, dir_name)
        os.makedirs(self.log_dir, exist_ok=True)
        if 'SLURM_JOB_ID' in os.environ:
            slurm_output_file = os.path.realpath(f'slurm_outputs/slurm-{os.environ["SLURM_JOB_ID"]}.out')
            if os.path.exists(slurm_output_file):
                try:
                    os.symlink(slurm_output_file, os.path.join(self.log_dir, os.path.basename(slurm_output_file)))
                except (OSError, FileExistsError):
                    pass

        self.write_to_tensorboard_events = write_to_tensorboard_events
        tb_dir = os.path.join(self.log_dir, 'tb')
        os.makedirs(tb_dir, exist_ok=True)

        self._csvs_dir = os.path.join(self.log_dir, 'csvs')
        os.makedirs(self._csvs_dir, exist_ok=True)

        self._figures_dir = os.path.join(self.log_dir, 'figures')
        os.makedirs(self._figures_dir, exist_ok=True)

        if on_dirs_created is not None:
            on_dirs_created(dict(
                group_dir=group_dir,
                log_dir=self.log_dir,
                tb_dir=tb_dir,
                csvs_dir=self._csvs_dir,
                figures_dir=self._figures_dir,
            ))

        logger_main.add_text_output(os.path.join(self.log_dir, 'debug.log'))
        logger_main.add_tabular_output(os.path.join(self.log_dir, 'progress.csv'))
        logger_per_frame.add_tabular_output(
            os.path.join(self.log_dir, 'progress_per_frame.csv.lz4'),
            mode='wt',
            open_func=LZ4FileWrapper.open,
        )
        # It looked that using gzip sometimes resulted in corrupted files.
        #logger_per_frame.add_tabular_output(
        #    os.path.join(self.log_dir, 'progress_per_frame.csv.gz'),
        #    mode='wt',
        #    open_func=GzipFileWrapper.open,
        #)
        logger_per_frame.set_log_tabular_only(True)
        logger_sub.add_tabular_output(os.path.join(self.log_dir, 'progress_gpi_details.csv'))

        git_commit = get_git_commit_hash()
        logger_main.log('Git commit: {}'.format(git_commit))
        #git_diff_file_path = os.path.join(self.log_dir, 'git_diff-{}.patch'.format(git_commit))
        #save_git_diff_to_file(git_diff_file_path)

        create_py_archive(os.path.join(self.log_dir, 'py_archive.tar.gz'))

        self._experiment = experiment
        self._loss = loss
        self._grad_norm = grad_norm

        super().__init__(log_dir=tb_dir, flush_secs=30)

    @property
    def figures_dir(self):
        return self._figures_dir

    def add_loss(self, name, value, step="frame", **kwargs):
        if self._loss:
            if torch.is_tensor(value):
                value = value.item()
            self.add_scalar("loss/" + name, value, step, **kwargs)

    def add_grad_norm(self, name, value, step="frame", **kwargs):
        if self._grad_norm:
            if torch.is_tensor(value):
                value = value.item()
            self.add_scalar("grad_norm/" + name, value, step, **kwargs)

    def add_evaluation(self, name, value, step="frame", prefix='evaluation/', **kwargs):
        self.add_scalar(prefix + name, value, self._get_step(step), **kwargs)

    def add_schedule(self, name, value, step="frame", **kwargs):
        if self._loss:
            self.add_scalar("schedule" + "/" + name, value, self._get_step(step), **kwargs)

    def add_scalar(self, name, value, step="frame", dump_targets=['episode'], write_to_tensorboard_events=False):
        '''
        Log an arbitrary scalar.
        Args:
            name (str): The tag to associate with the scalar
            value (number): The value of the scalar at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''
        if self.write_to_tensorboard_events or write_to_tensorboard_events:
            #super().add_scalar(self.env_name + "/" + name, value, self._get_step(step))
            super().add_scalar(name, value, self._get_step(step))
        if 'episode' in dump_targets:
            logger_main.record_tabular(name, value)
        if 'frame' in dump_targets:
            logger_per_frame.record_tabular(name, value)

    def add_string(self, name, value, step="frame", dump_targets=['episode']):
        if 'episode' in dump_targets:
            logger_main.record_tabular(name, value)
        if 'frame' in dump_targets:
            logger_per_frame.record_tabular(name, value)

    def add_summary(self, name, mean, std, max, min, step="frame", eval_prefix='evaluation/', **kwargs):
        self.add_evaluation(name + "/mean", mean, step, prefix=eval_prefix, **kwargs)
        self.add_evaluation(name + "/std", std, step, prefix=eval_prefix, **kwargs)
        self.add_evaluation(name + "/max", max, step, prefix=eval_prefix, **kwargs)
        self.add_evaluation(name + "/min", min, step, prefix=eval_prefix, **kwargs)

        # This is slow as hell.
        ####with open(os.path.join(self.log_dir, self.env_name, name + ".csv"), "a") as csvfile:
        ###with open(os.path.join(self._csvs_dir, name + ".csv"), "a") as csvfile:
        ###    csv.writer(csvfile).writerow([self._get_step(step), mean, std, max, min])

    def flush_csv(self, **kwargs):
        logger_main.dump_tabular(with_prefix=False, with_timestamp=False, **kwargs)

    def flush_csv_per_frame(self, **kwargs):
        logger_per_frame.dump_tabular(with_prefix=False, with_timestamp=False, **kwargs)

    def _get_step(self, _type):
        if _type == "frame":
            return self._experiment.frame
        if _type == "episode":
            return self._experiment.episode
        return _type

    def close(self):
        pass


def get_commit_hash():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False
    )
    return result.stdout.decode("utf-8").rstrip()


COMMIT_HASH = get_commit_hash()


def get_git_commit_hash():
    import subprocess
    p = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit, _ = p.communicate()
    git_commit = git_commit.strip().decode('utf-8')
    return git_commit

def save_git_diff_to_file(git_diff_file_path):
    import subprocess
    git_diff_file = open(git_diff_file_path, 'w')
    p = subprocess.Popen(['git', 'diff', '--patch', 'HEAD'], stdout=git_diff_file)
    p.wait()

def get_py_files():
    import subprocess
    p = subprocess.Popen(['git', 'ls-files', '--cached', '--others', '*.py'], stdout=subprocess.PIPE)
    py_files, _ = p.communicate()
    py_files = py_files.strip().decode('utf-8')
    return py_files.split('\n')

def create_py_archive(archive_file_path):
    p = subprocess.Popen(['tar', '-zcf', archive_file_path] + get_py_files())
    p.wait()

