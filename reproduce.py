"""Reproduce the statistics reported in the article from the summary tables in tables/.

Usage:  python3 reproduce.py        (requires numpy, pandas and scipy)

Each published statistic is printed next to the value computed from the tables. A value matches when it
is within one unit of the last digit printed in the article. The script exits with status 1 if any
statistic differs, other than the known differences listed in KNOWN_DIFFERENCES (explained in README.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TABLES = Path(__file__).resolve().parent / 'tables'
KNOWN_DIFFERENCES = {
    'Table 2: Others n': 'printed as 18; the count is 140 (the printed 28.1% is 140 of 499 selections)',
    'Table 3: AVE EE (from printed loadings)': 'printed AVE is higher than the printed loadings give',
    'Table 3: AVE FCA (from printed loadings)': 'printed AVE is higher than the printed loadings give',
    'Table 3: AVE LV (from printed loadings)': 'printed AVE is higher than the printed loadings give',
    'Section 5.3: use-case answers, min length': 'printed as 17 characters; the shortest answer has 24',
}
results = []


def report(label, published, value):
    """published is the article's value as printed ('0.246', '28.1') or a threshold ('p<0.001')."""
    if published.startswith('p'):
        ok = value < float(published[2:]) if published[1] == '<' else value > float(published[2:])
        shown = f'p = {value:.4f}'
    else:
        decimals = len(published.split('.')[1]) if '.' in published else 0
        ok = abs(value - float(published)) < 10 ** -decimals
        shown = f'{value:.{decimals + 1}f}'
    status = 'match' if ok else ('known difference' if label in KNOWN_DIFFERENCES else 'DIFFERS')
    results.append((label, published, shown, status))


def frequencies(table, variable, published):
    """published = [(n, percent), ...] in table order; percentages use the sample size as denominator."""
    rows = table[table['variable'] == variable]
    total = rows['n_published'].sum()
    for (_, row), (n, pct) in zip(rows.iterrows(), published):
        report(f'{variable}: {row["category"]} n', n, row['n_published'])
        report(f'{variable}: {row["category"]} %', pct, 100 * row['n_published'] / total)


def ols_from_moments(cov, x, y, n):
    """OLS coefficients, t values and p values from a covariance matrix (n - 1 denominator)."""
    sxx, sxy = cov.loc[x, x].values, cov.loc[x, y].values
    b = np.linalg.solve(sxx, sxy)
    df = n - len(x) - 1
    s2 = (n - 1) * (cov.loc[y, y] - b @ sxy) / df
    se = np.sqrt(np.diag(s2 * np.linalg.inv((n - 1) * sxx)))
    return b, b / se, 2 * stats.t.sf(np.abs(b / se), df)


# ---------------------------------------------------------------------------
# students (N = 377)
# ---------------------------------------------------------------------------

sf = pd.read_csv(TABLES / 'student_frequencies.csv')
N = int(sf.loc[sf['variable'] == 'Q2 age', 'n_published'].sum())
report('Section 4.3: student N', '377', N)
frequencies(sf, 'Q2 age', [('209', '55.4'), ('108', '28.6'), ('43', '11.4'), ('13', '3.4'), ('3', '0.8'),
                           ('1', '0.3')])
frequencies(sf, 'Q3 gender', [('121', '32.1'), ('231', '61.2'), ('16', '4.2'), ('8', '2.1'), ('1', '0.3')])
frequencies(sf, 'Q4 education', [('57', '15.1'), ('51', '13.5'), ('108', '28.6'), ('117', '31.0'), ('38', '10.1'),
                                 ('6', '1.6')])
frequencies(sf, 'Q5 Gen-AI usage', [('8', '2.1'), ('127', '33.7'), ('168', '44.6'), ('49', '13.0'), ('25', '6.6')])

at = pd.read_csv(TABLES / 'student_assignment_types.csv').set_index('option')['n_selections']
for option, label, n, pct in [
        ('Generating written content (essays, reports, etc.)', 'Written content', '183', '36.7'),
        ('Creating visuals or graphics', 'Visuals or graphics', '71', '14.2'),
        ('Language translation', 'Language translation', '50', '10.0'),
        ('Coding using Programming Languages (Python, Java, etc.)', 'Coding', '39', '7.8'),
        ('I am aware about the tools but have never used it', 'Aware but never used', '16', '3.2'),
        ('Others (please specify)', 'Others', '18', '28.1')]:
    report(f'Table 2: {label} n', n, at[option])
    report(f'Table 2: {label} % of selections', pct, 100 * at[option] / at.sum())

items = pd.read_csv(TABLES / 'student_items.csv')
cov_items = pd.read_csv(TABLES / 'student_item_covariance.csv', index_col='item')
constructs = {c: list(items.loc[items['construct'] == c, 'item']) for c in
              ['PE', 'EE', 'SI', 'FCA', 'HM', 'HA', 'LV', 'EIL', 'BI']}
for c, published in {'PE': '0.894', 'EE': '0.797', 'SI': '0.708', 'FCA': '0.628', 'LV': '0.838', 'EIL': '0.902',
                     'HM': '0.872', 'HA': '0.861', 'BI': '0.903'}.items():
    sub = cov_items.loc[constructs[c], constructs[c]].values
    k = len(sub)
    report(f'Table 3: alpha {c}', published, k / (k - 1) * (1 - np.trace(sub) / sub.sum()))

kept = sum(constructs.values(), [])
R = cov_items.loc[kept, kept].values
R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
inv = np.linalg.inv(R)
partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
off = ~np.eye(len(R), dtype=bool)
report('Table 3: KMO', '0.947', (R[off] ** 2).sum() / ((R[off] ** 2).sum() + (partial[off] ** 2).sum()))
p = len(R)
report('Table 3: Bartlett chi-square', '8521.247', -(N - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(R)[1])
report('Table 3: Bartlett df', '496', p * (p - 1) / 2)

# the loadings come from a factor analysis the article doesn't specify; AVE and CR follow from them
loadings = {'PE': '0.717 0.798 0.815 0.714', 'EE': '0.618 0.608 0.721 0.694', 'SI': '0.774 0.621 0.807',
            'FCA': '0.653 0.650 0.719', 'LV': '0.665 0.689 0.683 0.760', 'EIL': '0.773 0.791 0.697 0.740',
            'HM': '0.769 0.702 0.799', 'HA': '0.666 0.847 0.872 0.875', 'BI': '0.783 0.773 0.744'}
ave_cr = {'PE': ('0.581', '0.847'), 'EE': ('0.501', '0.756'), 'SI': ('0.545', '0.780'), 'FCA': ('0.495', '0.714'),
          'LV': ('0.500', '0.793'), 'EIL': ('0.565', '0.838'), 'HM': ('0.574', '0.801'), 'HA': ('0.672', '0.890'),
          'BI': ('0.588', '0.811')}
for c, printed in loadings.items():
    lam = np.array(printed.split(), dtype=float)
    report(f'Table 3: AVE {c} (from printed loadings)', ave_cr[c][0], (lam ** 2).mean())
    report(f'Table 3: CR {c} (from printed loadings)', ave_cr[c][1],
           lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum()))

# construct scores are item means, so their covariances follow from the item covariances
names = list(constructs)
A = pd.DataFrame(0.0, index=names, columns=cov_items.index)
for c, cols in constructs.items():
    A.loc[c, cols] = 1 / len(cols)
cov_c = A @ cov_items @ A.T
corr_c = cov_c / np.sqrt(np.outer(np.diag(cov_c), np.diag(cov_c)))
order = ['PE', 'EE', 'LV', 'EIL', 'HM', 'HA', 'SI', 'BI', 'FCA']
table4 = ['0.518', '0.769 0.447', '0.714 0.440 0.787', '0.635 0.451 0.723 0.675', '0.611 0.455 0.604 0.650 0.605',
          '0.305 0.230 0.264 0.285 0.262 0.227', '0.691 0.460 0.641 0.691 0.671 0.707 0.339',
          '0.361 0.436 0.366 0.295 0.351 0.356 0.352 0.383']
for i, row in enumerate(table4, 1):
    for j, published in enumerate(row.split()):
        a, b = order[i], order[j]
        report(f'Table 4: r({a.replace("HA", "HABIT")}, {b.replace("HA", "HABIT")})', published, corr_c.loc[a, b])

predictors = ['PE', 'EE', 'FCA', 'SI', 'HA', 'HM', 'LV', 'EIL']
rxx, rxy = corr_c.loc[predictors, predictors].values, corr_c.loc[predictors, 'BI'].values
beta = np.linalg.solve(rxx, rxy)
r2 = beta @ rxy
df_resid = N - len(predictors) - 1
report('Section 5.1.4: regression F(8, 368)', '91.353', (r2 / 8) / ((1 - r2) / df_resid))
report('Section 5.1.4: R-squared', '0.665', r2)
report('Section 5.1.4: adjusted R-squared', '0.658', 1 - (1 - r2) * (N - 1) / df_resid)
se = np.sqrt((1 - r2) / df_resid * np.diag(np.linalg.inv(rxx)))
p_beta = 2 * stats.t.sf(np.abs(beta / se), df_resid)
for (h, c, published, threshold), b, pv in zip(
        [('H1', 'PE', '0.246', 'p<0.001'), ('H2', 'EE', '0.006', 'p>0.05'), ('H3', 'FCA', '0.05', 'p>0.05'),
         ('H4', 'SI', '0.086', 'p<0.05'), ('H5', 'HA', '0.315', 'p<0.001'), ('H6', 'HM', '0.217', 'p<0.001'),
         ('H7', 'LV', '-0.091', 'p>0.05'), ('H8', 'EIL', '0.194', 'p<0.001')], beta, p_beta):
    report(f'Table 5: {h} {c} beta', published, b)
    report(f'Table 5: {h} {c} p', threshold, pv)

mm = pd.read_csv(TABLES / 'student_moderation_moments.csv', index_col='variable')
cov_m = mm.drop(columns='mean')
for h, c, b_pub, t_pub, threshold in [
        ('H9a', 'PE', '-0.101', '-2.441', 'p<0.05'), ('H9b', 'EE', '-0.066', None, 'p>0.05'),
        ('H9c', 'FCA', '-0.1774', '-3.367', 'p<0.001'), ('H9d', 'SI', '-0.033', None, 'p>0.05'),
        ('H9e', 'HA', '-0.1284', '-3.351', 'p<0.001'), ('H9f', 'HM', '-0.1174', '-2.896', 'p<0.05'),
        ('H10a', 'LV', '-0.149', '-2.399', 'p<0.05'), ('H10b', 'EIL', '-0.112', None, 'p>0.05')]:
    b, t, pv = ols_from_moments(cov_m, [c, 'usage', f'{c}_x_usage'], 'BI', N)  # PROCESS model 1
    report(f'Table 5: {h} {c} x usage b', b_pub, b[2])
    if t_pub:
        report(f'Table 5: {h} {c} x usage t(373)', t_pub, t[2])
    report(f'Table 5: {h} {c} x usage p', threshold, pv[2])

# ---------------------------------------------------------------------------
# faculty (N = 62)
# ---------------------------------------------------------------------------

ff = pd.read_csv(TABLES / 'faculty_frequencies.csv')
report('Section 4.3: faculty N', '62', ff.loc[ff['variable'] == 'Q31 gender', 'n_published'].sum())
frequencies(ff, 'Q31 gender', [('17', '27.4'), ('42', '67.8'), ('1', '1.6'), ('2', '3.2')])
frequencies(ff, 'Q21 ethnicity', [('1', '1.6'), ('1', '1.6'), ('6', '9.7'), ('52', '83.8'), ('2', '3.2')])
frequencies(ff, 'Q18 years teaching', [('5', '8.1'), ('6', '9.7'), ('15', '24.2'), ('36', '58.1')])
frequencies(ff, 'Q19.1 class size', [('1', '1.6'), ('4', '6.5'), ('12', '19.4'), ('26', '42'), ('10', '16.1'),
                                     ('9', '14.5')])
for variable, figure, published in [('Teaching approach', 'Fig 8', ['3.2', '1.6', '54.8', '25.8', '14.5']),
                                    ('Gen AI Integration', 'Fig 9', ['1.6', '9.7', '24.2', '29.1', '12.9', '22.6'])]:
    rows = ff[ff['variable'] == variable]
    for (_, row), pct in zip(rows.iterrows(), published):
        report(f'{figure}: {row["category"]} %', pct, 100 * row['n_published'] / rows['n_published'].sum())

tp = pd.read_csv(TABLES / 'faculty_tp_items.csv').set_index('item')
tp_means = '3.15 3.44 3.32 3.05 3.67 2.77 3.37 3.93 3.3 3.28 3.53 2.53 2.44 2.98 3.33 4.07'.split()
tp_sds = '0.705 1.000 1.055 1.245 1.139 1.053 1.029 0.776 0.886 0.996 0.984 0.868 1.102 1.217 1.024 0.884'.split()
for i, (m, s) in enumerate(zip(tp_means, tp_sds), 1):
    report(f'Table 7: TP{i} mean', m, tp.loc[f'TP{i}', 'mean'])
    report(f'Table 7: TP{i} SD', s, tp.loc[f'TP{i}', 'sd'])

# Spearman correlations: missing values were replaced with the unrounded series mean of Qualtrics' codes
ct = pd.read_csv(TABLES / 'faculty_integration_crosstabs.csv')
for h, variable, rs_pub, threshold in [('H11', 'Teaching approach', '0.026', 'p>0.05'),
                                       ('H12', 'Gen AI usage', '0.499', 'p<0.001')]:
    rows = ct[ct['variable'] == variable]
    x = np.repeat(rows['code'].values, rows['n'].values).astype(float)
    y = np.repeat(rows['integration_code'].values, rows['n'].values).astype(float)
    x[np.isnan(x)], y[np.isnan(y)] = np.nanmean(x), np.nanmean(y)
    rs, pv = stats.spearmanr(x, y)
    report(f'Table 5: {h} rs(62)', rs_pub, rs)
    report(f'Table 5: {h} p', threshold, pv)

# ---------------------------------------------------------------------------
# faculty open-ended responses (clustering)
# ---------------------------------------------------------------------------

lengths = pd.read_csv(TABLES / 'cluster_response_lengths.csv').set_index('question')
for question, n in [('use_cases', '57'), ('barriers', '55'), ('concerns', '52'), ('perceptions', '59')]:
    report(f'Section 5.3: {question} responses', n, lengths.loc[question, 'n_responses'])
for stat, published in [('min_chars', '17'), ('max_chars', '577'), ('mean_chars', '170.2'), ('median_chars', '121')]:
    report(f'Section 5.3: use-case answers, {stat.split("_")[0]} length', published, lengths.loc['use_cases', stat])

cc = pd.read_csv(TABLES / 'cluster_integration_counts.csv')
sizes = cc.groupby(['question', 'figure_label'])['n'].sum()
for figure, question, published in [
        ('Fig 11', 'use_cases', {'Assignment assistance, Course content creation, Educational tool': '30',
                                 'Creativity enhancement, Educational tool': '19', 'Idea generation, Teaching aid': '8'}),
        ('Fig 13', 'barriers', {'Fear, lack of knowledge, cost': '21', 'Academic integrity, Lack of understanding': '13',
                                'Academic integrity, Need for training, Reliability': '9', 'Cost, Learning curve': '6',
                                'Learning curve, Need for training': '6'}),
        ('Fig 15', 'concerns', {'Academic integrity, Ethical implications, Misinformation': '23',
                                'Academic integrity, Misuse and overreliance': '15',
                                'Ethical implications, Misuse and overreliance, Loss of originality': '10',
                                'Academic integrity, Lack of understanding': '4'}),
        ('Fig 17', 'perceptions', {'Advantage, Time-saving, Tool for Learning': '25', 'Ambivalence, Uncertainty': '15',
                                   'Advantage with caution, Ethical use, Creativity enhancement': '12',
                                   'Overall positive perception': '7'})]:
    for label, n in published.items():
        report(f'{figure}: {label}', n, sizes.loc[(question, label)])
educational = sizes.loc['use_cases'][sizes.loc['use_cases'].index.str.contains('Educational tool')].sum()
report('Section 5.3: use-case answers coded "Educational tool"', '49', educational)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

width = max(len(r[0]) for r in results)
for label, published, shown, status in results:
    note = f'  ({KNOWN_DIFFERENCES[label]})' if status == 'known difference' else ''
    print(f'{label:{width}s}  published {published:>9s}  reproduced {shown:>12s}  {status}{note}')
counts = pd.Series([r[3] for r in results]).value_counts()
print(f'\n{len(results)} statistics: ' + ', '.join(f'{n} {s}' for s, n in counts.items()))
unused = set(KNOWN_DIFFERENCES) - {r[0] for r in results if r[3] == 'known difference'}
if unused:
    print('Known differences that now match (update KNOWN_DIFFERENCES):', sorted(unused))
print('\nNot printed in the article, so not compared: the cluster means and confidence intervals in Figs 12, 14,\n'
      '16 and 18 (cluster_integration_counts.csv gives every value plotted) and the simple slopes in Figs 3-7\n'
      '(computable from student_moderation_moments.csv).')
sys.exit(1 if 'DIFFERS' in counts else 0)
