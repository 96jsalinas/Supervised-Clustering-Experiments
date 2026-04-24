library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(knitr)

metrics_path <- "figures/dashboard_full_comparison/metrics_table.csv"
df <- read_csv(metrics_path, show_col_types = FALSE)

df <- df %>%
  mutate(
    combo     = str_split_fixed(run, "__", 3)[, 3],
    generator = str_split_fixed(dataset_tag, "_", 2)[, 1],
    base_tag  = str_replace(dataset_tag, "_s[0-9]+$", ""),
    axis      = str_split_fixed(dataset_tag, "_", 3)[, 2]
  )

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
