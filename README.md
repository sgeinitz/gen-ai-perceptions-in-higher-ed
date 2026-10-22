# Faculty and Student Perceptions of Generative AI in the Classroom: Data and Code to Reproduce Findings

This repository contains partially aggregated data from the two surveys in
described in the paper, "*Exploring the Pedagogical Potential: An Investigation into Faculty and Students' Perceptions of Integrating Generative AI in the Classroom*", and that was conducted at our university in the Fall semester of 2024.
Together with `reproduce.py`, they reproduce the statistics reported in the
article. No individual responses are included.

**Citation:** article/DOI  TBD 

## Data availability

Summary-level data from both surveys are publicly available in this repository along with code that reproduces the statistics reported in the article. These data comprise frequency tables, item-level means, standard deviations and covariance matrices, and cross-tabulations; no individual responses are included. In accordance with the informed consent approved by our university's Institutional Review Board, under which participants were told that only the research team would have access to their responses, individual-level data are not publicly available. Anonymized individual-level data, with all identifying information removed, are available from the corresponding author on reasonable request, subject to approval by the university Institutional Review Board.

## Quick start

```
python3 reproduce.py
```

This needs Python 3 with numpy, pandas and scipy. It prints every published statistic next to the value computed from the tables: 269 of 274 match, and the other 5 are the known differences listed below. A value matches when it is within one unit of the last digit printed in the article.

## Tables

| File | Contents | Reproduces |
|---|---|---|
| `tables/student_frequencies.csv` | Age, gender, education level and Gen-AI usage (N = 377) | Tables 1–2 |
| `tables/student_assignment_types.csv` | Selections per assignment type (multi-select; 499 selections) | Table 2 |
| `tables/student_items.csv` | The 34 agreement items: construct, wording, n answered, mean, SD | Table 3 |
| `tables/student_item_covariance.csv` | Item covariance matrix | Tables 3–5: alphas, KMO, Bartlett's test, construct correlations, regression |
| `tables/student_moderation_moments.csv` | Means and covariances of the construct scores, Gen-AI usage and their products | Table 5 moderation; Figs 3–7 |
| `tables/faculty_frequencies.csv` | Gender, ethnicity, years teaching, class size, teaching approach, integration level (N = 62) | Table 6; Figs 8–9 |
| `tables/faculty_tp_items.csv` | The 16 teacher-perception items: wording, reverse coding, n, mean, SD | Table 7 |
| `tables/faculty_integration_crosstabs.csv` | Teaching approach × integration level and Gen-AI usage × integration level | H11, H12 |
| `tables/cluster_integration_counts.csv` | Integration level counts per cluster for the four open-ended questions | Figs 11–18 |
| `tables/cluster_response_lengths.csv` | Number and length (characters) of the open-ended answers | Section 5.3 |

## How the tables were made

- **Samples:**
  - Students (N = 377): non-spam responses that answered "Disagree" on both attention-check items.
  - Faculty (N = 62): 57 respondents plus 5 blank responses whose values were all filled in by mean imputation, as in the article.
- **Coding:**
  - Student agreement items are 7-point, coded 1 (Strongly disagree) to 7 (Strongly agree), except the LV and EIL items, which had five options and are coded 1 to 5 (the `scale` column in `student_items.csv`).
  - Gen-AI usage is coded 1 (Never) to 5 (Very frequently).
  - Faculty teacher-perception items are coded 1 to 5, with TP2, TP4, TP11, TP14 and TP16 reverse-coded.
  - The faculty cross-tabs use Qualtrics' numeric codes, which skip some values.
- **Construct scores** are item means. `student_items.csv` lists each item's construct; two items were dropped, one each from SI and FCA.
- **Missing values** were replaced with the series mean, as in the article's SPSS analysis.
  - **Frequency tables** count each filled-in value in the category of the rounded mean. They show `n_answered`, `n_missing_filled` and `n_published` (the sum of the two).
  - **Covariances, moments and the correlations** use the unrounded mean. Covariances use an n − 1 denominator.
  - **Teacher-perception item statistics** use the answers given.

## Privacy

- **Only aggregates are included.** No table crosses demographic variables, and the demographic frequencies are one-way and identical to Tables 1 and 6 of the article.
- **Some small cells are kept.** The cross-tabs of attitudes and the cluster counts are complete, including cells of 1 or 2, so the correlations and cluster results reproduce exactly. These tables carry no demographics or answer text.
- **Individual-level data:** available on request only; see [Data availability](#data-availability).



## Not reproducible from these tables

- **The clustering itself and Fig 10 (t-SNE):** these need the answer text and its sentence embeddings.
- **Confidence intervals in Figs 12, 14, 16 and 18:** these were bootstrapped. The counts give the exact cluster means and allow the intervals to be recomputed.
- **Cluster labels:** `figure_label` gives the labels printed in the article. For three barriers clusters, these differ from the saved `summary` text, but the cluster sizes and means are identical.

## Survey instruments

The item wording in `tables/student_items.csv` and `tables/faculty_tp_items.csv`, and the integration-stage labels in `tables/faculty_frequencies.csv`, are adapted from published instruments. These remain the work of their original authors, so please cite them when reusing the items:

- **Student constructs PE, EE, SI, FCA, HM, HA and BI:** Venkatesh et al. (2012).
- **Learning value (LV):** Ain et al. (2015).
- **Empowerment in learning (EIL):** Cacciamani et al. (2018).
- **Faculty teacher-perception items and integration stages:** Wozney et al. (2006), as refined by Khukalenko et al. (2022).

References:

- Ain, N., Kaur, K., & Waheed, M. (2015). The influence of learning value on learning management system use: An extension of UTAUT2. *Information Development, 32*(5), 1306–1321.
- Cacciamani, S., Villani, D., Bonanomi, A., Carissoli, C., Olivari, M. G., Morganti, L., & Confalonieri, E. (2018). Factors affecting students' acceptance of tablet PCs: A study in Italian high schools. *Journal of Research on Technology in Education, 50*(2), 120–133.
- Khukalenko, I., Kaplan-Rakowski, R., & An, Y. (2022). Teachers' perceptions of using virtual reality technology in classrooms: A large-scale survey. *SSRN Electronic Journal.* https://doi.org/10.2139/ssrn.4074142
- Venkatesh, V., Thong, J. Y., & Xu, X. (2012). Consumer acceptance and use of information technology: Extending the unified theory of acceptance and use of technology. *MIS Quarterly, 36*(1), 157–178.
- Wozney, L., Venkatesh, V., & Abrami, P. (2006). Implementing computer technologies: Teachers' perceptions and practices. *Journal of Technology and Teacher Education, 14*(1), 173–207.

## License

- **Data and documentation** (the tables in `tables/` and this README): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt them for any purpose, provided you give appropriate credit; please cite the article (see Citation).
- **Code** (`reproduce.py`): MIT License.

The full texts of both licenses are in [`LICENSE`](LICENSE). The licenses cover this repository's data and code, not the original survey instruments (see Survey instruments).
