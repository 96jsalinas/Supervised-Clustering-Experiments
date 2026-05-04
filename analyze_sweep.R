library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(knitr)
library(ggplot2)

# Inputs / outputs. Override via positional args:
#   Rscript analyze_sweep.R <metrics_csv> <figures_dir>
args <- commandArgs(trailingOnly = TRUE)
metrics_path <- if (length(args) >= 1) args[[1]] else "figures/dashboard_full_comparison/metrics_table.csv"
figures_dir  <- if (length(args) >= 2) args[[2]] else dirname(metrics_path)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(metrics_path, show_col_types = FALSE)

df <- df %>%
  mutate(
    combo        = str_split_fixed(run, "__", 3)[, 3],
    generator    = str_split_fixed(dataset_tag, "_", 2)[, 1],
    base_tag     = str_replace(dataset_tag, "_s[0-9]+$", ""),
    axis         = str_split_fixed(dataset_tag, "_", 3)[, 2],
    seed         = str_extract(dataset_tag, "s[0-9]+$"),
    attribution  = str_split_fixed(combo, "_", 3)[, 1],
    dr           = str_split_fixed(combo, "_", 3)[, 2],
    clusterer    = str_split_fixed(combo, "_", 3)[, 3]
  )

# ---------------------------------------------------------------------------
# Tables (as before)
# ---------------------------------------------------------------------------

cat("\n===== 1. Mean classifier accuracy per generator =====\n")
acc_by_gen <- df %>%
  distinct(dataset_tag, model_tag, generator, classifier_accuracy) %>%
  group_by(generator, model_tag) %>%
  summarise(mean_acc = mean(classifier_accuracy), .groups = "drop") %>%
  pivot_wider(names_from = model_tag, values_from = mean_acc) %>%
  mutate(gap = lightgbm - mlp)
print(kable(acc_by_gen, digits = 4))

cat("\n===== 2. Cross-cell classifier gap distribution =====\n")
gap_by_cell <- df %>%
  distinct(dataset_tag, model_tag, classifier_accuracy) %>%
  pivot_wider(names_from = model_tag, values_from = classifier_accuracy) %>%
  mutate(gap = lightgbm - mlp)
print(summary(gap_by_cell$gap))
cat(sprintf(
  "cells with MLP >= LightGBM: %d of %d (%.1f%%)\n",
  sum(gap_by_cell$gap <= 0),
  nrow(gap_by_cell),
  100 * mean(gap_by_cell$gap <= 0)
))

cat("\n===== 3. Classifier accuracy by base cell =====\n")
acc_by_base <- df %>%
  distinct(dataset_tag, model_tag, base_tag, classifier_accuracy) %>%
  group_by(base_tag, model_tag) %>%
  summarise(mean_acc = mean(classifier_accuracy), .groups = "drop") %>%
  pivot_wider(names_from = model_tag, values_from = mean_acc) %>%
  mutate(gap = lightgbm - mlp)
print(kable(acc_by_base, digits = 3))

cat("\n===== 4. Seed variance of classifier accuracy per base cell =====\n")
seed_var <- df %>%
  distinct(dataset_tag, model_tag, base_tag, classifier_accuracy) %>%
  group_by(base_tag, model_tag) %>%
  summarise(
    mean_acc = mean(classifier_accuracy),
    sd_acc   = sd(classifier_accuracy),
    n_seeds  = n(),
    .groups  = "drop"
  )
print(kable(seed_var, digits = 3))
cat(sprintf(
  "sd range across (base_tag x model): %.3f to %.3f\n",
  min(seed_var$sd_acc), max(seed_var$sd_acc)
))

cat("\n===== 5. Best ARI per (dataset, model), averaged by axis x generator =====\n")
best_ari <- df %>%
  group_by(dataset_tag, model_tag, generator, axis) %>%
  summarise(best_ari = max(ari), .groups = "drop") %>%
  group_by(generator, axis, model_tag) %>%
  summarise(mean_ari = mean(best_ari), .groups = "drop") %>%
  pivot_wider(names_from = model_tag, values_from = mean_ari) %>%
  mutate(gap = lightgbm - mlp) %>%
  arrange(generator, axis)
print(kable(best_ari, digits = 3))

cat("\n===== 6. Mean ARI by model x combo (across all 42 cells, both spaces) =====\n")
combo_rank <- df %>%
  group_by(model_tag, combo) %>%
  summarise(mean_ari = mean(ari), .groups = "drop") %>%
  pivot_wider(names_from = model_tag, values_from = mean_ari) %>%
  arrange(desc((lightgbm + mlp) / 2))
print(kable(combo_rank, digits = 3))

cat("\n===== 7. HDBSCAN noise and collapse breakdown =====\n")
noise_check <- df %>%
  filter(str_detect(combo, "hdbscan")) %>%
  group_by(model_tag, combo, space) %>%
  summarise(
    median_n_noise    = median(n_noise),
    mean_n_noise      = mean(n_noise),
    median_n_clusters = median(n_clusters),
    full_collapse     = sum(n_noise == 1000),
    n_runs            = n(),
    .groups           = "drop"
  ) %>%
  arrange(model_tag, combo, space)
print(kable(noise_check, digits = 1))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

theme_set(theme_minimal(base_size = 11))

save_plot <- function(p, name, w = 8, h = 5) {
  path <- file.path(figures_dir, name)
  ggsave(path, p, width = w, height = h, dpi = 150)
  cat(sprintf("wrote %s\n", path))
}

# Long-form ARI dataframe filtered to the embedded 2D space (where the
# clustering decision is actually made) for the ARI grouped boxplots.
ari_emb <- df %>% filter(space == "embedding_2d")

# ARI by model (faceted by space so we keep the full-attribution view too).
p_ari_model <- ggplot(df, aes(x = model_tag, y = ari, fill = model_tag)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  facet_wrap(~ space) +
  labs(title = "ARI by model", x = NULL, y = "ARI") +
  theme(legend.position = "none")
save_plot(p_ari_model, "fig_ari_by_model.png")

# ARI by DR method, faceted by clusterer (embedded 2D only).
p_ari_dr <- ggplot(ari_emb, aes(x = dr, y = ari, fill = dr)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  facet_grid(model_tag ~ clusterer) +
  labs(title = "ARI by DR method (embedded 2D)", x = NULL, y = "ARI") +
  theme(legend.position = "none")
save_plot(p_ari_dr, "fig_ari_by_dr.png", w = 9, h = 6)

# ARI by clusterer, faceted by DR.
p_ari_clust <- ggplot(ari_emb, aes(x = clusterer, y = ari, fill = clusterer)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  facet_grid(model_tag ~ dr) +
  labs(title = "ARI by clusterer (embedded 2D)", x = NULL, y = "ARI") +
  theme(legend.position = "none")
save_plot(p_ari_clust, "fig_ari_by_clusterer.png", w = 9, h = 6)

# ARI by attribution method (will be more interesting once Sweep B is in).
if (length(unique(df$attribution)) > 1) {
  p_ari_attr <- ggplot(ari_emb, aes(x = attribution, y = ari, fill = attribution)) +
    geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
    facet_grid(model_tag ~ dr + clusterer) +
    labs(title = "ARI by attribution method (embedded 2D)", x = NULL, y = "ARI") +
    theme(legend.position = "none")
  save_plot(p_ari_attr, "fig_ari_by_attribution.png", w = 11, h = 6)
}

# ARI by base cell (one boxplot per dataset axis, seeds + combos as the
# distribution). Embedded 2D only to keep the panel readable.
p_ari_base <- ggplot(ari_emb, aes(x = base_tag, y = ari, fill = model_tag)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  labs(title = "ARI by base cell (embedded 2D)", x = NULL, y = "ARI", fill = "model") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p_ari_base, "fig_ari_by_base_cell.png", w = 11, h = 5)

# Classifier accuracy by base cell x model. One row per (dataset_tag, model)
# so seeds form the distribution.
acc_long <- df %>% distinct(dataset_tag, base_tag, model_tag, classifier_accuracy)
p_acc_base <- ggplot(acc_long, aes(x = base_tag, y = classifier_accuracy, fill = model_tag)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  labs(title = "Classifier accuracy by base cell", x = NULL, y = "accuracy", fill = "model") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p_acc_base, "fig_accuracy_by_base_cell.png", w = 11, h = 5)

# Timings: long-form across the four pipeline stages, log-y so cheap and
# expensive stages stay readable in the same panel.
time_long <- df %>%
  select(model_tag, attribution, dr, clusterer, combo, space,
         time_model_fit, time_attribution, time_reduction, time_clustering) %>%
  pivot_longer(starts_with("time_"), names_to = "stage", values_to = "seconds") %>%
  mutate(stage = str_remove(stage, "^time_"))

p_time_model <- ggplot(time_long, aes(x = stage, y = seconds, fill = model_tag)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  scale_y_log10() +
  labs(title = "Pipeline stage timings by model", x = NULL, y = "seconds (log scale)", fill = "model")
save_plot(p_time_model, "fig_timings_by_model.png", w = 9, h = 5)

p_time_combo <- ggplot(time_long, aes(x = combo, y = seconds, fill = model_tag)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  facet_wrap(~ stage, scales = "free_y") +
  scale_y_log10() +
  labs(title = "Pipeline stage timings by method combo", x = NULL, y = "seconds (log scale)", fill = "model") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p_time_combo, "fig_timings_by_combo.png", w = 12, h = 7)

# Seed variability: ARI distribution per base cell x model x combo, with
# seeds forming the spread. Highlights how ruido-dominado each cell is.
p_seed_spread <- ari_emb %>%
  ggplot(aes(x = base_tag, y = ari, colour = model_tag)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.6,
               position = position_dodge(width = 0.8)) +
  facet_wrap(~ combo) +
  labs(title = "Seed-level ARI spread per base cell (embedded 2D)",
       x = NULL, y = "ARI", colour = "model") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p_seed_spread, "fig_seed_spread_ari.png", w = 13, h = 8)

# ---------------------------------------------------------------------------
# F-measure panels (pair-counting, Larsen & Aone 1999) — mirror the ARI set.
# ---------------------------------------------------------------------------

if ("f_measure" %in% names(df)) {
  fm_emb <- df %>% filter(space == "embedding_2d")

  p_fm_model <- ggplot(df, aes(x = model_tag, y = f_measure, fill = model_tag)) +
    geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
    facet_wrap(~ space) +
    labs(title = "F-measure (pair-counting) by model", x = NULL, y = "F-measure") +
    theme(legend.position = "none")
  save_plot(p_fm_model, "fig_fmeasure_by_model.png")

  p_fm_dr <- ggplot(fm_emb, aes(x = dr, y = f_measure, fill = dr)) +
    geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
    facet_grid(model_tag ~ clusterer) +
    labs(title = "F-measure (pair-counting) by DR method (embedded 2D)", x = NULL, y = "F-measure") +
    theme(legend.position = "none")
  save_plot(p_fm_dr, "fig_fmeasure_by_dr.png", w = 9, h = 6)

  p_fm_clust <- ggplot(fm_emb, aes(x = clusterer, y = f_measure, fill = clusterer)) +
    geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
    facet_grid(model_tag ~ dr) +
    labs(title = "F-measure (pair-counting) by clusterer (embedded 2D)", x = NULL, y = "F-measure") +
    theme(legend.position = "none")
  save_plot(p_fm_clust, "fig_fmeasure_by_clusterer.png", w = 9, h = 6)

  p_fm_base <- ggplot(fm_emb, aes(x = base_tag, y = f_measure, fill = model_tag)) +
    geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
    labs(title = "F-measure (pair-counting) by base cell (embedded 2D)",
         x = NULL, y = "F-measure", fill = "model") +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  save_plot(p_fm_base, "fig_fmeasure_by_base_cell.png", w = 11, h = 5)
}

# ---------------------------------------------------------------------------
# Attribution timing and cost-benefit figures (Sweep B: LRP vs SHAP).
# Self-activate when more than one attribution method is present.
# ---------------------------------------------------------------------------

if (length(unique(df$attribution)) > 1) {
  # Boxplot of attribution wall time by method, faceted by base cell.
  p_time_attr <- ggplot(
    df %>% filter(space == "embedding_2d"),
    aes(x = attribution, y = time_attribution, fill = attribution)
  ) +
    geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
    facet_wrap(~ base_tag, scales = "free_y") +
    scale_y_log10() +
    labs(title = "Attribution wall time: LRP vs SHAP by data cell",
         x = NULL, y = "seconds (log scale)") +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 25, hjust = 1))
  save_plot(p_time_attr, "fig_timing_attribution.png", w = 13, h = 8)

  # Cost-benefit scatter: ARI vs time_attribution, coloured by attribution
  # method. Fixed to the most informative combo (umap + kmeans) to keep the
  # primary panel readable; the full multi-panel version can be generated
  # separately if needed.
  cb_data <- df %>%
    filter(space == "embedding_2d", dr == "umap", clusterer == "kmeans")
  if (nrow(cb_data) > 0) {
    p_cost_benefit <- ggplot(
      cb_data,
      aes(x = time_attribution, y = ari, colour = attribution)
    ) +
      geom_point(alpha = 0.7, size = 1.8) +
      geom_abline(slope = 0, intercept = 0, linetype = "dashed",
                  colour = "grey60", linewidth = 0.4) +
      scale_x_log10() +
      facet_wrap(~ model_tag) +
      labs(title = "ARI vs attribution wall time (UMAP + k-means, embedded 2D)",
           x = "time_attribution (seconds, log scale)", y = "ARI",
           colour = "attribution") +
      theme(legend.position = "bottom")
    save_plot(p_cost_benefit, "fig_ari_vs_time_attribution.png", w = 10, h = 5)
  }
}

cat("\nDone. Figures written to: ", figures_dir, "\n", sep = "")
