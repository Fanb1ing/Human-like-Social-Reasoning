#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subgroup-aware dynamic data selection for the social-reasoning GRPO pipeline."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from accelerate.utils import gather_object
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainerCallback
from trl import GRPOTrainer


DEMOGRAPHIC_PATTERNS = {
    "gender": r"您的性别：([^\n]+)",
    "age": r"您的年龄段：([^\n]+)",
    "education": r"您的学历：([^\n]+)",
    "occupation": r"您目前从事的职业：([^\n]+)",
    "income": r"您的年收入情况：([^\n]+)",
}
DEMOGRAPHIC_COLUMNS = tuple(DEMOGRAPHIC_PATTERNS)

# comparison_data.csv uses English labels while the training prompts use the
# Chinese labels below. Multiple China-benchmark income bins intentionally map
# into the same training bin. ``None`` means that the benchmark category has no
# representable training category; it is excluded and the remaining China
# distribution is renormalized before calibration.
CHINA_BENCHMARK_GROUP_MAP: dict[str, dict[str, str | None]] = {
    "gender": {
        "Female": "女",
        "Male": "男",
    },
    "age": {
        "18-27": "18~27",
        "28-40": "28~40",
        "41-50": "41~50",
        "51-60": "51~60",
        "60+": "60以上",
    },
    "education": {
        "Bachelor's Degree": "大学本科",
        "Master's Degree or Above": "研究生及以上",
        "High School/Technical School": "高中/中专",
        "College": "大学专科",
        "Middle School or Below": "初中及以下",
    },
    "occupation": {
        "Student": "学生",
        "Teacher": "老师",
        "Tech Developer/Engineer": "技术开发/工程师",
        "Freelancer": "自由职业",
        "Worker/Laborer": "工人劳动者",
        "Administrative": "行政",
        "Government Official": "党政机关人员",
        "Medical Staff": "医护人员",
        "Service Worker": "服务业人员",
        "Self-employed": "个体经营者",
        "Marketing/Sales/Business": "市场/销售/商务",
        "Business Manager": "企业管理者",
        "Finance/Accounting/Audit": "财务/会计/出纳/审计",
        "Researcher": "科研人员",
        "Designer": "设计从业者",
        "Retired": "离休/退休",
        "Agriculture/Fishery Worker": "农林牧渔劳动者",
        "Other / Hard to Map": None,
    },
    "income": {
        "Low quintile": "1万元以下",
        "Lower-middle quintile": "1万元以下",
        "Middle quintile": "1万元至10万元",
        "Upper-middle quintile": "1万元至10万元",
        "High quintile": "10万元至50万元",
    },
}


def add_selection_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a stable row-level sample ID and parse the five persona attributes."""
    if "prompt" not in dataframe.columns:
        raise ValueError("The dataframe must contain a 'prompt' column.")

    result = dataframe.copy()
    if "sample_id" not in result.columns:
        result.insert(0, "sample_id", np.arange(len(result), dtype=np.int64))
    if result["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique for every CSV row.")

    for column, pattern in DEMOGRAPHIC_PATTERNS.items():
        result[column] = result["prompt"].str.extract(pattern, expand=False).str.strip()
        missing = result[column].isna()
        if missing.any():
            examples = result.loc[missing, "sample_id"].head(5).tolist()
            raise ValueError(f"Failed to parse {column} for sample IDs: {examples}")
    return result


def compute_smoothed_group_targets(
    human_reference: pd.DataFrame,
    eta: float = 0.2,
) -> dict[str, dict[str, float]]:
    """Compute human-distribution targets with a small uniform smoothing term."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}.")

    targets: dict[str, dict[str, float]] = {}
    for feature in DEMOGRAPHIC_COLUMNS:
        if feature not in human_reference.columns:
            raise ValueError(f"Missing human-reference feature: {feature}")
        counts = human_reference[feature].value_counts(dropna=False)
        if counts.empty or counts.index.isna().any():
            raise ValueError(f"Invalid human-reference values for {feature}.")
        human_ratio = counts / counts.sum()
        uniform = 1.0 / len(counts)
        targets[feature] = {
            str(group): float((1.0 - eta) * ratio + eta * uniform)
            for group, ratio in human_ratio.items()
        }
        total = sum(targets[feature].values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"Smoothed targets for {feature} sum to {total}, not 1.")
    return targets


def compute_dataset_group_proportions(
    dataset_reference: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return marginal demographic proportions from the full training data."""
    proportions: dict[str, dict[str, float]] = {}
    for feature in DEMOGRAPHIC_COLUMNS:
        if feature not in dataset_reference.columns:
            raise ValueError(f"Missing training-data feature: {feature}")
        counts = dataset_reference[feature].value_counts(dropna=False)
        if counts.empty or counts.index.isna().any():
            raise ValueError(f"Invalid training-data values for {feature}.")
        ratios = counts / counts.sum()
        proportions[feature] = {
            str(group): float(ratio) for group, ratio in ratios.items()
        }
    return proportions


def load_aligned_china_benchmark_proportions(
    comparison_data: pd.DataFrame,
    dataset_proportions: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Map China-benchmark labels to training labels and renormalize each axis."""
    required = {"chart", "side", "item", "percentage"}
    missing = required.difference(comparison_data.columns)
    if missing:
        raise ValueError(f"comparison_data is missing columns: {sorted(missing)}")

    china_rows = comparison_data.loc[
        comparison_data["side"].astype(str) == "china_benchmark"
    ].copy()
    if china_rows.empty:
        raise ValueError("comparison_data has no china_benchmark rows.")
    if china_rows["percentage"].isna().any():
        raise ValueError("China-benchmark percentages contain NaN.")
    if (china_rows["percentage"] < 0).any():
        raise ValueError("China-benchmark percentages must be non-negative.")

    aligned: dict[str, dict[str, float]] = {}
    for feature in DEMOGRAPHIC_COLUMNS:
        mapping = CHINA_BENCHMARK_GROUP_MAP[feature]
        feature_rows = china_rows.loc[china_rows["chart"] == feature]
        observed_items = set(feature_rows["item"].astype(str))
        unknown_items = observed_items.difference(mapping)
        missing_items = set(mapping).difference(observed_items)
        if unknown_items or missing_items:
            raise ValueError(
                f"China-benchmark mapping mismatch for {feature}: "
                f"unknown={sorted(unknown_items)}, missing={sorted(missing_items)}"
            )

        accumulated: Counter[str] = Counter()
        for row in feature_rows.itertuples(index=False):
            mapped_group = mapping[str(row.item)]
            if mapped_group is not None:
                accumulated[mapped_group] += float(row.percentage)
        total = float(sum(accumulated.values()))
        if total <= 0:
            raise ValueError(f"No representable China groups for {feature}.")
        china_feature = {
            str(group): float(value / total)
            for group, value in accumulated.items()
        }

        dataset_groups = set(map(str, dataset_proportions[feature]))
        china_groups = set(china_feature)
        if china_groups != dataset_groups:
            raise ValueError(
                f"Mapped China groups do not match training groups for {feature}: "
                f"China-only={sorted(china_groups - dataset_groups)}, "
                f"data-only={sorted(dataset_groups - china_groups)}"
            )
        aligned[feature] = china_feature
    return aligned


def compute_dataset_china_difference_lower_bounds(
    dataset_reference: pd.DataFrame,
    comparison_data: pd.DataFrame,
    alpha: float = 0.2,
) -> dict[str, dict[str, float]]:
    """Compute the user's demographic lower-bound mass from a China difference.

    For every demographic group g:

        l_g = (1 - alpha) * p_data,g
              + alpha * (p_china,g - p_data,g)
            = (1 - 2*alpha) * p_data,g + alpha * p_china,g

    With alpha=0.2 the lower bounds sum to 0.8 on every demographic axis,
    deliberately leaving roughly 20% of K for value-driven selection.
    """
    if not 0.0 <= alpha <= 0.5:
        raise ValueError(f"alpha must be in [0, 0.5], got {alpha}.")
    dataset_proportions = compute_dataset_group_proportions(dataset_reference)
    china_proportions = load_aligned_china_benchmark_proportions(
        comparison_data,
        dataset_proportions,
    )

    lower_bounds: dict[str, dict[str, float]] = {}
    for feature in DEMOGRAPHIC_COLUMNS:
        lower_bounds[feature] = {
            group: float(
                (1.0 - alpha) * data_ratio
                + alpha * (china_proportions[feature][group] - data_ratio)
            )
            for group, data_ratio in dataset_proportions[feature].items()
        }
        total = sum(lower_bounds[feature].values())
        if not math.isclose(
            total,
            1.0 - alpha,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                f"Calibrated lower bounds for {feature} sum to {total}, "
                f"not {1.0 - alpha}."
            )
        if any(value < 0.0 for value in lower_bounds[feature].values()):
            raise AssertionError(f"Negative calibrated lower bound for {feature}.")
    return lower_bounds


def compute_group_quotas(
    selection_size: int,
    group_targets: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, int]]:
    """Convert target proportions into per-round marginal lower bounds."""
    if selection_size <= 0:
        raise ValueError("selection_size must be positive.")
    return {
        feature: {
            str(group): int(math.floor(selection_size * float(target)))
            for group, target in targets.items()
        }
        for feature, targets in group_targets.items()
    }


def compute_capacity_capped_group_quotas(
    selection_size: int,
    group_targets: Mapping[str, Mapping[str, float]],
    metadata: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Cap desired lower bounds by the unique rows available in full training data."""
    desired = compute_group_quotas(selection_size, group_targets)
    available = {
        feature: metadata[feature].astype(str).value_counts().to_dict()
        for feature in DEMOGRAPHIC_COLUMNS
    }
    capped: dict[str, dict[str, int]] = {}
    for feature, feature_quotas in desired.items():
        capped[feature] = {}
        for group, quota in feature_quotas.items():
            if str(group) not in available[feature]:
                raise ValueError(f"Training data has no rows for {feature}={group}.")
            capped[feature][str(group)] = min(
                int(quota),
                int(available[feature][str(group)]),
            )
    return capped


def count_subgroups(
    sample_ids: Iterable[int],
    metadata: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Count marginal demographic membership for a collection of sample IDs."""
    indexed = metadata.set_index("sample_id", drop=False)
    counts: dict[str, dict[str, int]] = {}
    ids = list(map(int, sample_ids))
    for feature in DEMOGRAPHIC_COLUMNS:
        counts[feature] = {
            str(group): int(count)
            for group, count in indexed.loc[ids, feature].value_counts().items()
        }
    return counts


def unmet_quotas(
    sample_ids: Iterable[int],
    metadata: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]],
) -> dict[tuple[str, str], int]:
    """Return positive quota deficits."""
    counts = count_subgroups(sample_ids, metadata)
    deficits: dict[tuple[str, str], int] = {}
    for feature, feature_quotas in quotas.items():
        for group, quota in feature_quotas.items():
            deficit = int(quota) - counts.get(feature, {}).get(str(group), 0)
            if deficit > 0:
                deficits[(feature, str(group))] = deficit
    return deficits


def compute_prompt_values(
    rollout_rewards: pd.DataFrame,
    num_generations: int,
    lambda_conflict: float = 1.0,
    reward_weights: Sequence[float] = (0.25, 0.25, 0.50),
) -> pd.DataFrame:
    """Aggregate per-rollout rewards into boundary, conflict, and final value."""
    required = {"sample_id", "process_reward", "rule_reward", "answer_reward"}
    missing = required.difference(rollout_rewards.columns)
    if missing:
        raise ValueError(f"Missing rollout reward columns: {sorted(missing)}")
    if num_generations < 2:
        raise ValueError("GRPO data value requires at least two generations.")
    if lambda_conflict < 0:
        raise ValueError("lambda_conflict must be non-negative.")
    if len(reward_weights) != 3:
        raise ValueError("reward_weights must be [process, rule, answer].")

    rewards = rollout_rewards.copy()
    reward_columns = ["process_reward", "rule_reward", "answer_reward"]
    if rewards[reward_columns].isna().any().any():
        raise ValueError("NaN found in rollout rewards.")
    tolerance = 1e-6
    if (
        (rewards[reward_columns].to_numpy() < -tolerance).any()
        or (rewards[reward_columns].to_numpy() > 1.0 + tolerance).any()
    ):
        raise ValueError("All selection rewards must be normalized to [0, 1].")

    group_sizes = rewards.groupby("sample_id", sort=False).size()
    bad_sizes = group_sizes[group_sizes != num_generations]
    if not bad_sizes.empty:
        preview = bad_sizes.head(10).to_dict()
        raise ValueError(
            f"Each sample must have exactly {num_generations} rollouts; got {preview}"
        )

    weights = np.asarray(reward_weights, dtype=np.float64)
    rewards["total_reward"] = rewards[reward_columns].to_numpy() @ weights
    rewards["rollout_conflict"] = (
        rewards["answer_reward"] - rewards["process_reward"]
    ).abs()

    grouped = rewards.groupby("sample_id", sort=False)
    values = grouped.agg(
        mean_total_reward=("total_reward", "mean"),
        mean_process_reward=("process_reward", "mean"),
        mean_rule_reward=("rule_reward", "mean"),
        mean_answer_reward=("answer_reward", "mean"),
        conflict_score=("rollout_conflict", "mean"),
    )
    centered = rewards["total_reward"] - rewards["sample_id"].map(
        values["mean_total_reward"]
    )
    rewards["absolute_deviation"] = centered.abs()
    values["reward_dispersion"] = grouped["absolute_deviation"].mean()
    values["boundary_score"] = (
        4.0
        * values["mean_total_reward"]
        * (1.0 - values["mean_total_reward"])
        * values["reward_dispersion"]
    )
    values["final_value"] = values["boundary_score"] * (
        1.0 + lambda_conflict * values["conflict_score"]
    )
    values = values.reset_index()
    return values


def trim_distributed_rollout_padding(
    rollout_rewards: pd.DataFrame,
    num_generations: int,
) -> pd.DataFrame:
    """Drop only the tail replicas added to equalize distributed eval batches."""
    if "sample_id" not in rollout_rewards:
        raise ValueError("Missing rollout reward column: sample_id")
    group_sizes = rollout_rewards.groupby("sample_id", sort=False).size()
    undersized = group_sizes[group_sizes < num_generations]
    if not undersized.empty:
        raise ValueError(
            "Candidate scoring produced too few rollouts; "
            f"got {undersized.head(10).to_dict()}"
        )
    occurrence = rollout_rewards.groupby("sample_id", sort=False).cumcount()
    return rollout_rewards.loc[occurrence < num_generations].reset_index(drop=True)


class DynamicSubsetDataset(TorchDataset):
    """Fixed-length view whose active rows can change without rebuilding Trainer."""

    def __init__(self, full_dataset: Any, active_ids: Sequence[int]):
        self.full_dataset = full_dataset
        self._active_ids: list[int] = []
        self.set_active_ids(active_ids)

    @property
    def active_ids(self) -> list[int]:
        return list(self._active_ids)

    def set_active_ids(self, active_ids: Sequence[int]) -> None:
        ids = [int(sample_id) for sample_id in active_ids]
        if not ids:
            raise ValueError("DynamicSubsetDataset cannot be empty.")
        if len(ids) != len(set(ids)):
            raise ValueError("active_ids must not contain duplicate sample IDs.")
        if min(ids) < 0 or max(ids) >= len(self.full_dataset):
            raise IndexError("active_ids contains a row outside the full dataset.")
        if self._active_ids and len(ids) != len(self._active_ids):
            raise ValueError(
                "The active subset length must remain fixed after Trainer creates its sampler."
            )
        self._active_ids = ids

    def __len__(self) -> int:
        return len(self._active_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.full_dataset[self._active_ids[index]]


def _candidate_capacity(
    candidate_ids: Sequence[int],
    metadata: pd.DataFrame,
) -> Counter[tuple[str, str]]:
    indexed = metadata.set_index("sample_id", drop=False)
    capacity: Counter[tuple[str, str]] = Counter()
    for sample_id in candidate_ids:
        row = indexed.loc[int(sample_id)]
        for feature in DEMOGRAPHIC_COLUMNS:
            capacity[(feature, str(row[feature]))] += 1
    return capacity


def ensure_candidate_quota_capacity(
    candidate_ids: Sequence[int],
    metadata: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]],
    seed: int,
) -> list[int]:
    """Expand a candidate pool until every marginal quota is individually feasible."""
    indexed = metadata.set_index("sample_id", drop=False)
    candidates = list(dict.fromkeys(map(int, candidate_ids)))
    candidate_set = set(candidates)
    rng = np.random.default_rng(seed)

    for feature, feature_quotas in quotas.items():
        for group, quota in feature_quotas.items():
            capacity = _candidate_capacity(candidates, metadata)[(feature, str(group))]
            missing = int(quota) - capacity
            if missing <= 0:
                continue
            available = indexed.index[
                (indexed[feature].astype(str) == str(group))
                & (~indexed.index.isin(candidate_set))
            ].to_numpy(dtype=np.int64)
            if len(available) < missing:
                raise RuntimeError(
                    f"Full data cannot provide quota {feature}={group}: "
                    f"need {quota}, available {capacity + len(available)}."
                )
            rng.shuffle(available)
            additions = available[:missing].tolist()
            candidates.extend(additions)
            candidate_set.update(additions)
    return candidates


def build_candidate_pool(
    metadata: pd.DataFrame,
    candidate_size: int,
    round_id: int,
    previous_selected: Sequence[int],
    last_scored_round: Mapping[int, int],
    quotas: Mapping[str, Mapping[str, int]],
    seed: int,
) -> list[int]:
    """Keep prior selected samples and rotate in the least-recently-scored rows."""
    all_ids = metadata["sample_id"].astype(int).tolist()
    if candidate_size < 0 or candidate_size >= len(all_ids):
        return all_ids
    if candidate_size <= 0:
        raise ValueError("candidate_size must be positive or -1 for the full pool.")

    rng = np.random.default_rng(seed + 104729 * int(round_id))
    previous = list(dict.fromkeys(map(int, previous_selected)))
    if len(previous) > candidate_size:
        previous = previous[:candidate_size]
    candidate_set = set(previous)

    remaining = [sample_id for sample_id in all_ids if sample_id not in candidate_set]
    tie_break = {sample_id: float(rng.random()) for sample_id in remaining}
    remaining.sort(
        key=lambda sample_id: (
            int(last_scored_round.get(sample_id, -1)),
            tie_break[sample_id],
        )
    )
    candidates = previous + remaining[: max(0, candidate_size - len(previous))]
    return ensure_candidate_quota_capacity(
        candidates,
        metadata=metadata,
        quotas=quotas,
        seed=seed + 1299709 * int(round_id),
    )


def _quota_counts_for_selection(
    selected: Sequence[int],
    indexed_metadata: pd.DataFrame,
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sample_id in selected:
        row = indexed_metadata.loc[int(sample_id)]
        for feature in DEMOGRAPHIC_COLUMNS:
            counts[(feature, str(row[feature]))] += 1
    return counts


def _repair_selection(
    selected: list[int],
    candidate_ids: Sequence[int],
    score_by_id: Mapping[int, float],
    indexed_metadata: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]],
) -> list[int]:
    """Repair residual quota deficits by value-aware swaps that preserve met quotas."""
    selected_set = set(selected)
    counts = _quota_counts_for_selection(selected, indexed_metadata)
    max_swaps = len(selected) * len(DEMOGRAPHIC_COLUMNS)

    for _ in range(max_swaps):
        deficits = {
            (feature, str(group)): int(quota) - counts[(feature, str(group))]
            for feature, feature_quotas in quotas.items()
            for group, quota in feature_quotas.items()
            if counts[(feature, str(group))] < int(quota)
        }
        if not deficits:
            return selected

        target = max(
            deficits,
            key=lambda key: deficits[key] / max(int(quotas[key[0]][key[1]]), 1),
        )
        target_feature, target_group = target
        incoming_options = [
            sample_id
            for sample_id in candidate_ids
            if sample_id not in selected_set
            and str(indexed_metadata.loc[sample_id, target_feature]) == target_group
        ]
        incoming_options.sort(key=lambda sid: (-score_by_id[sid], sid))
        outgoing_options = sorted(selected, key=lambda sid: (score_by_id[sid], sid))

        swapped = False
        for incoming in incoming_options:
            incoming_row = indexed_metadata.loc[incoming]
            for outgoing in outgoing_options:
                outgoing_row = indexed_metadata.loc[outgoing]
                safe = True
                for feature in DEMOGRAPHIC_COLUMNS:
                    outgoing_group = str(outgoing_row[feature])
                    incoming_group = str(incoming_row[feature])
                    updated = counts[(feature, outgoing_group)] - 1
                    if incoming_group == outgoing_group:
                        updated += 1
                    required = int(quotas.get(feature, {}).get(outgoing_group, 0))
                    if updated < required:
                        safe = False
                        break
                if not safe:
                    continue

                selected[selected.index(outgoing)] = incoming
                selected_set.remove(outgoing)
                selected_set.add(incoming)
                for feature in DEMOGRAPHIC_COLUMNS:
                    counts[(feature, str(outgoing_row[feature]))] -= 1
                    counts[(feature, str(incoming_row[feature]))] += 1
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            break

    # Local swaps can become stuck because each row participates in five
    # simultaneous marginal constraints. Fall back to a small exact binary
    # program over the current candidate pool rather than silently violating a
    # quota. This path has only K*2 variables in the default configuration.
    return _exact_quota_select(
        candidate_ids=candidate_ids,
        selection_size=len(selected),
        score_by_id=score_by_id,
        indexed_metadata=indexed_metadata,
        quotas=quotas,
    )


def _exact_quota_select(
    candidate_ids: Sequence[int],
    selection_size: int,
    score_by_id: Mapping[int, float],
    indexed_metadata: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]],
) -> list[int]:
    """Exact fallback: maximize value under all marginal lower bounds."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    ids = list(map(int, candidate_ids))
    rows = [np.ones(len(ids), dtype=np.float64)]
    lower = [float(selection_size)]
    upper = [float(selection_size)]

    for feature, feature_quotas in quotas.items():
        feature_values = indexed_metadata.loc[ids, feature].astype(str).to_numpy()
        for group, quota in feature_quotas.items():
            rows.append((feature_values == str(group)).astype(np.float64))
            lower.append(float(quota))
            upper.append(np.inf)

    constraint_matrix = np.vstack(rows)
    # A tiny deterministic tie-break keeps repeated runs stable without
    # materially changing the learned data-value objective.
    values = np.asarray([score_by_id[sample_id] for sample_id in ids], dtype=np.float64)
    tie_break = np.arange(len(ids), dtype=np.float64) * 1e-12
    result = milp(
        c=-values + tie_break,
        integrality=np.ones(len(ids), dtype=np.int8),
        bounds=Bounds(
            lb=np.zeros(len(ids), dtype=np.float64),
            ub=np.ones(len(ids), dtype=np.float64),
        ),
        constraints=LinearConstraint(
            constraint_matrix,
            lb=np.asarray(lower, dtype=np.float64),
            ub=np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "Candidate pool has no jointly feasible subgroup selection: "
            f"status={result.status}, message={result.message}"
        )
    selected = [sample_id for sample_id, keep in zip(ids, result.x) if keep > 0.5]
    if len(selected) != selection_size:
        raise RuntimeError(
            f"Exact quota fallback returned {len(selected)} rows, expected {selection_size}."
        )
    return selected


def subgroup_aware_greedy_select(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]],
    selection_size: int,
    gamma: float = 1.0,
) -> list[int]:
    """Maximize current data value while enforcing marginal subgroup lower bounds."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if scores["sample_id"].duplicated().any():
        raise ValueError("Scores must contain one row per sample_id.")
    if len(scores) < selection_size:
        raise ValueError(
            f"Candidate pool has {len(scores)} rows, smaller than K={selection_size}."
        )

    indexed_metadata = metadata.set_index("sample_id", drop=False)
    candidate_ids = scores["sample_id"].astype(int).tolist()
    missing_metadata = set(candidate_ids).difference(indexed_metadata.index)
    if missing_metadata:
        raise ValueError(f"Missing metadata for sample IDs: {sorted(missing_metadata)[:10]}")

    score_by_id = {
        int(row.sample_id): float(row.final_value)
        for row in scores[["sample_id", "final_value"]].itertuples(index=False)
    }
    selected: list[int] = []
    remaining = set(candidate_ids)
    counts: Counter[tuple[str, str]] = Counter()

    while len(selected) < selection_size:
        remaining_slots = selection_size - len(selected)
        deficits = {
            (feature, str(group)): max(
                int(quota) - counts[(feature, str(group))],
                0,
            )
            for feature, feature_quotas in quotas.items()
            for group, quota in feature_quotas.items()
        }
        mandatory_features = {
            feature
            for feature in quotas
            if sum(
                deficits[(feature, str(group))]
                for group in quotas[feature]
            )
            >= remaining_slots
        }

        ranked: list[tuple[float, float, float, int]] = []
        fallback: list[tuple[float, float, int]] = []
        for sample_id in remaining:
            row = indexed_metadata.loc[sample_id]
            coverage = 0.0
            covers_mandatory = True
            for feature in DEMOGRAPHIC_COLUMNS:
                group = str(row[feature])
                quota = int(quotas.get(feature, {}).get(group, 0))
                deficit = deficits.get((feature, group), 0)
                if deficit > 0:
                    coverage += deficit / max(quota, 1)
                if feature in mandatory_features and deficit <= 0:
                    covers_mandatory = False
            value = score_by_id[sample_id]
            combined = value * (1.0 + gamma * coverage)
            fallback.append((coverage, value, -sample_id))
            if covers_mandatory:
                ranked.append((combined, coverage, value, -sample_id))

        if ranked:
            chosen = -max(ranked)[3]
        else:
            chosen = -max(fallback)[2]

        selected.append(chosen)
        remaining.remove(chosen)
        chosen_row = indexed_metadata.loc[chosen]
        for feature in DEMOGRAPHIC_COLUMNS:
            counts[(feature, str(chosen_row[feature]))] += 1

    if unmet_quotas(selected, metadata, quotas):
        selected = _repair_selection(
            selected,
            candidate_ids=candidate_ids,
            score_by_id=score_by_id,
            indexed_metadata=indexed_metadata,
            quotas=quotas,
        )

    deficits = unmet_quotas(selected, metadata, quotas)
    if deficits:
        raise AssertionError(f"Subgroup quotas remain unmet: {deficits}")
    if len(selected) != selection_size or len(set(selected)) != selection_size:
        raise AssertionError("Selection size or uniqueness invariant failed.")
    return selected


class SelectionAwareGRPOTrainer(GRPOTrainer):
    """Thin GRPOTrainer extension that captures existing per-function rewards."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._selection_capture_enabled = False
        self._selection_rollout_records: list[dict[str, float | int]] = []
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        # TRL's generation context temporarily disables gradient checkpointing
        # and re-enables it without the configured kwargs. Restore the original
        # non-reentrant mode required by LoRA + DDP after every generation call.
        unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
        was_gradient_checkpointing = unwrapped_model.is_gradient_checkpointing
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            if was_gradient_checkpointing:
                unwrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=(
                        self.args.gradient_checkpointing_kwargs or {}
                    )
                )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(
            inputs,
            prompts,
            completions,
            completion_ids_list,
        )
        if self._selection_capture_enabled:
            if len(self.reward_funcs) != 3:
                raise RuntimeError(
                    "Dynamic selection expects reward order [process, rule, answer]."
                )
            local_ids = [int(example["sample_id"]) for example in inputs]
            all_ids = gather_object(local_ids)
            reward_rows = rewards_per_func.detach().float().cpu().tolist()
            if len(all_ids) != len(reward_rows):
                raise RuntimeError(
                    f"Gathered {len(all_ids)} sample IDs but {len(reward_rows)} rewards."
                )
            self._selection_rollout_records.extend(
                {
                    "sample_id": int(sample_id),
                    "process_reward": float(row[0]),
                    "rule_reward": float(row[1]),
                    "answer_reward": float(row[2]),
                }
                for sample_id, row in zip(all_ids, reward_rows)
            )
        return rewards_per_func

    def score_candidates(
        self,
        candidate_dataset: Any,
        num_generations: int,
        lambda_conflict: float,
    ) -> pd.DataFrame:
        """Generate current-policy rollouts and return one value row per candidate."""
        self.accelerator.wait_for_everyone()
        self._selection_rollout_records = []
        self._selection_capture_enabled = True

        was_training = self.model.training
        self.model_wrapped.eval()
        try:
            dataloader = self.get_eval_dataloader(candidate_dataset)
            with torch.inference_mode():
                for generation_batch in dataloader:
                    generated = self._generate_and_score_completions(generation_batch)
                    del generated
        finally:
            self._selection_capture_enabled = False
            if was_training:
                self.model_wrapped.train()
            self._metrics["eval"].clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()

        rollout_rewards = trim_distributed_rollout_padding(
            pd.DataFrame(self._selection_rollout_records),
            num_generations=num_generations,
        )
        values = compute_prompt_values(
            rollout_rewards,
            num_generations=num_generations,
            lambda_conflict=lambda_conflict,
            reward_weights=self.reward_weights.detach().float().cpu().tolist(),
        )
        expected = set(map(int, candidate_dataset["sample_id"]))
        observed = set(values["sample_id"].astype(int))
        if expected != observed:
            missing = sorted(expected.difference(observed))[:10]
            extra = sorted(observed.difference(expected))[:10]
            raise RuntimeError(
                f"Candidate scoring ID mismatch; missing={missing}, extra={extra}"
            )
        return values


class SubgroupAwareDynamicSelector:
    """Stateful candidate rotation, scoring, constrained selection, and logging."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        group_targets: Mapping[str, Mapping[str, float]],
        selection_size: int,
        candidate_size: int,
        output_dir: str | os.PathLike[str],
        num_generations: int = 4,
        lambda_conflict: float = 1.0,
        gamma: float = 1.0,
        seed: int = 42,
        resume_state: bool = False,
    ):
        self.metadata = metadata.copy()
        self.selection_size = int(selection_size)
        self.candidate_size = int(candidate_size)
        self.num_generations = int(num_generations)
        self.lambda_conflict = float(lambda_conflict)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.quotas = compute_capacity_capped_group_quotas(
            self.selection_size,
            group_targets,
            self.metadata,
        )
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "selection_logs"
        self.state_path = self.output_dir / "selection_state.json"
        self.previous_selected: list[int] = []
        self.last_scored_round: dict[int, int] = {}
        self.last_round = -1
        if resume_state and self.state_path.exists():
            self._load_state()

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.previous_selected = [int(x) for x in state["previous_selected"]]
        self.last_scored_round = {
            int(sample_id): int(round_id)
            for sample_id, round_id in state["last_scored_round"].items()
        }
        self.last_round = int(state["last_round"])
        if len(self.previous_selected) != self.selection_size:
            raise ValueError(
                "Saved selection size does not match the current configuration."
            )

    def initial_active_ids(self) -> list[int]:
        if self.previous_selected:
            return list(self.previous_selected)
        return self.metadata["sample_id"].astype(int).head(self.selection_size).tolist()

    def refresh(
        self,
        trainer: SelectionAwareGRPOTrainer,
        full_dataset: Any,
        round_id: int,
        global_step: int,
    ) -> list[int]:
        if round_id <= self.last_round and self.previous_selected:
            return list(self.previous_selected)

        candidate_ids = build_candidate_pool(
            metadata=self.metadata,
            candidate_size=self.candidate_size,
            round_id=round_id,
            previous_selected=self.previous_selected,
            last_scored_round=self.last_scored_round,
            quotas=self.quotas,
            seed=self.seed,
        )
        candidate_dataset = full_dataset.select(candidate_ids)
        scores = trainer.score_candidates(
            candidate_dataset,
            num_generations=self.num_generations,
            lambda_conflict=self.lambda_conflict,
        )
        for sample_id in candidate_ids:
            self.last_scored_round[int(sample_id)] = int(round_id)

        selected = subgroup_aware_greedy_select(
            scores=scores,
            metadata=self.metadata,
            quotas=self.quotas,
            selection_size=self.selection_size,
            gamma=self.gamma,
        )
        self.previous_selected = selected
        self.last_round = int(round_id)

        if trainer.accelerator.is_main_process:
            self._write_round_log(
                scores=scores,
                candidate_ids=candidate_ids,
                selected=selected,
                round_id=round_id,
                global_step=global_step,
            )
            self._save_state(global_step=global_step)
        trainer.accelerator.wait_for_everyone()
        return list(selected)

    def _write_round_log(
        self,
        scores: pd.DataFrame,
        candidate_ids: Sequence[int],
        selected: Sequence[int],
        round_id: int,
        global_step: int,
    ) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        selected_set = set(map(int, selected))
        detail = scores.merge(
            self.metadata[
                ["sample_id", *DEMOGRAPHIC_COLUMNS, "question_index"]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        detail["selected"] = detail["sample_id"].astype(int).isin(selected_set)
        detail["selection_round"] = int(round_id)
        detail["global_step"] = int(global_step)
        detail.to_csv(
            self.log_dir / f"round_{round_id:04d}_scores.csv",
            index=False,
        )

        selected_counts = count_subgroups(selected, self.metadata)
        summary = {
            "round_id": int(round_id),
            "global_step": int(global_step),
            "candidate_count": len(candidate_ids),
            "selection_count": len(selected),
            "candidate_mean_value": float(scores["final_value"].mean()),
            "selected_mean_value": float(
                scores.loc[
                    scores["sample_id"].astype(int).isin(selected_set),
                    "final_value",
                ].mean()
            ),
            "quotas": self.quotas,
            "selected_subgroup_counts": selected_counts,
        }
        (self.log_dir / f"round_{round_id:04d}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_state(self, global_step: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "last_round": self.last_round,
            "global_step_when_selected": int(global_step),
            "previous_selected": self.previous_selected,
            "last_scored_round": {
                str(sample_id): round_id
                for sample_id, round_id in self.last_scored_round.items()
            },
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)


class DynamicSelectionCallback(TrainerCallback):
    """Refresh the mutable training subset at each selection-sized epoch."""

    def __init__(
        self,
        trainer: SelectionAwareGRPOTrainer,
        selector: SubgroupAwareDynamicSelector,
        dynamic_dataset: DynamicSubsetDataset,
        full_dataset: Any,
        selection_interval: int,
    ):
        self.trainer = trainer
        self.selector = selector
        self.dynamic_dataset = dynamic_dataset
        self.full_dataset = full_dataset
        self.selection_interval = int(selection_interval)
        self._last_refresh_step: int | None = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._last_refresh_step == int(state.global_step):
            return control
        if int(state.global_step) % self.selection_interval != 0:
            raise RuntimeError(
                f"Selection epoch began at global_step={state.global_step}, "
                f"not a multiple of interval={self.selection_interval}."
            )
        round_id = int(state.global_step) // self.selection_interval
        selected = self.selector.refresh(
            trainer=self.trainer,
            full_dataset=self.full_dataset,
            round_id=round_id,
            global_step=int(state.global_step),
        )
        self.dynamic_dataset.set_active_ids(selected)
        self._last_refresh_step = int(state.global_step)
        return control
