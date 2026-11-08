"""Plot eval win/draw/loss rates vs each bot from a run's log.csv.

  python3 ocean/aiants/train/plot.py runs/poc [more runs...] --out curves.png
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BOTS = ['random', 'gatherer', 'raider']
# best scripted bot's win rate vs each opponent (train.py --baselines, 200 games)
REF = {'random': 0.70, 'gatherer': 0.41, 'raider': 0.43}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default='curves.png')
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for run in a.runs:
        rows = [r for r in csv.DictReader(open(run + '/log.csv')) if r.get('eval_random_win')]
        steps = [int(r['step']) / 1e6 for r in rows]
        for ax, bot in zip(axes, BOTS):
            ax.plot(steps, [float(r['eval_%s_win' % bot]) for r in rows], label=run + ' win')
            ax.plot(steps, [float(r['eval_%s_loss' % bot]) for r in rows], '--',
                label=run + ' loss', alpha=0.7)
    for ax, bot in zip(axes, BOTS):
        ax.axhline(REF[bot], color='gray', lw=0.8, ls=':', label='best scripted bot')
        ax.set_title('vs %s bot' % bot)
        ax.set_xlabel('training steps (M)')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('fraction of eval games')
    axes[-1].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
