import json
import matplotlib.pyplot as plt
import numpy as np

# ── Load results and compute accuracies from source (no hardcoded integers) ───
def load(fn):
    d = json.load(open(fn))
    return d if isinstance(d, list) else list(d.values())

acc = load('results/accuracy_results_200_clean.json')
err = load('results/error_annotations_200.json')
err_by_id = {r['id']: r.get('error_types', []) for r in err}
N = len(acc)

def correct(r, c):
    return r[f'pred_{c}'] == r['correct_answer']

def pct(cond, subset=None):
    rows = subset if subset is not None else acc
    n = sum(correct(r, cond) for r in rows)
    return 100.0 * n / len(rows)

# C3b lives in a separate file
c3b_data = load('results/accuracy_results_c3b_200.json')
_c3b_pred = [k for k in c3b_data[0] if 'pred' in k][0]
c3b_pct = 100.0 * sum(r[_c3b_pred] == r['correct_answer'] for r in c3b_data) / len(c3b_data)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
})

BLUE   = '#1a4e8a'   # C1
RED    = '#c0392b'   # C2
ORANGE = '#e67e22'   # C3
TEAL   = '#b8e186'   # C3b
PURPLE = '#7d3c98'   # C4
GREEN  = '#1e8449'   # C5

# ── Figure 1: Main accuracy ───────────────────────────────────────────────────
labels = ['C1', 'C2', 'C3', 'C3b', 'C4', 'C5']
values = [pct('c1'), pct('c2'), pct('c3'), c3b_pct, pct('c4'), pct('c5')]
colors = [BLUE, RED, ORANGE, TEAL, PURPLE, GREEN]
c1_val = values[0]

fig, ax = plt.subplots(figsize=(6.5, 2.7))
x = np.arange(len(labels))
bars = ax.bar(x, values, width=0.55, color=colors, zorder=3,
              edgecolor='white', linewidth=0.6)

# 95% Wilson interval (n = 200), used as error bars
def wilson(p, n=N, z=1.96):
    p = p / 100.0
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)
ci = [wilson(v) for v in values]
yerr = np.array([[v - lo for v, (lo, hi) in zip(values, ci)], [hi - v for v, (lo, hi) in zip(values, ci)]])
ax.errorbar(x, values, yerr=yerr, fmt='none', ecolor='#333333', elinewidth=0.8, capsize=2.5, zorder=4)

# Reference line for C1
ax.axhline(c1_val, color=BLUE, linestyle=':', linewidth=1.2, alpha=0.7, zorder=2)

# Value labels (.1f)
for bar, val, (lo, hi) in zip(bars, values, ci):
    ax.text(bar.get_x() + bar.get_width() / 2, hi + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

descs = ['High-quality\ntranslation', 'MT Baseline',
         'GPT-4o-mini\ncorrection', 'Terminology-only\ncorrection',
         'DeepSeek\ncorrection', 'GPT-5.5\ncorrection']
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n{d}' for l, d in zip(labels, descs)], fontsize=7.5)

ax.set_ylabel('Accuracy (%)', fontsize=9)
ax.set_ylim(0, 72)
ax.set_xlim(-0.5, 5.5)

plt.tight_layout()
plt.savefig('paper/figures/fig1_accuracy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/fig1_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Figure 2: Error type impact (unchanged: integer labels) ───────────────────
groups = ['T1\n(n=191)', 'T2\n(n=77)', 'T3\n(n=168)', 'All three\n(n=53)']
c2_vals = [34, 26, 33, 25]
c3_vals = [39, 35, 38, 34]
c5_vals = [43, 36, 41, 32]

x = np.arange(len(groups))
width = 0.22

fig, ax = plt.subplots(figsize=(6.5, 3.4))
b1 = ax.bar(x - width, c2_vals, width, label='C2 (MT Baseline)',
            color=RED, zorder=3, edgecolor='white', linewidth=0.5)
b2 = ax.bar(x, c3_vals, width, label='C3 (GPT-4o-mini)',
            color=ORANGE, zorder=3, edgecolor='white', linewidth=0.5)
b3 = ax.bar(x + width, c5_vals, width, label='C5 (GPT-5.5)',
            color=GREEN, zorder=3, edgecolor='white', linewidth=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                str(int(h)), ha='center', va='bottom',
                fontsize=7.2, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel('Accuracy (%)', fontsize=9)
ax.set_ylim(0, 92)
ax.set_xlim(-0.5, 4.5)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.22),
          framealpha=0.9, ncol=3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('paper/figures/fig2_error_types.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/fig2_error_types.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Print Figure 1 values for cross-checking against the manuscript ───────────
print("Figure 1 (from data, .1f):",
      {l: f'{v:.1f}' for l, v in zip(labels, values)})
print("Figure 2: unchanged integer labels (C2/C3/C5):",
      c2_vals, c3_vals, c5_vals)