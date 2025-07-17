import os
import re
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Set, Optional, Counter
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.helpers import setup_logging, load_json, save_json

# Initialize logging
logger = setup_logging()


class ContentPatterns:
    """Defines patterns for identifying problematic content in text samples."""
    MISMATCH = [
        "CONTENT_MISMATCH",
        "CONTENT MISMATCH",
        "[CONTENT_MISMATCH]",
        "content_mismatch",
        "content mismatch"
    ]

    PLACEHOLDER = [
        r"^\[.+\]$",         # Strings that are entirely within square brackets
        r"^Unanswerable$",
        r"^No answers$",
        r"^[A-E]$",          # Single letter responses
        r"^\w{1,3}$"       # Very short responses (1-3 characters)
    ]


class MetricAssertion:
    """Represents a single metric condition for filtering."""
    def __init__(self, metric_name, rule=None):
        self.metric_name = metric_name
        self.rule = rule
        self.operator = None
        self.threshold = None

    def isGreaterThan(self, threshold):
        self.operator = ">"
        self.threshold = threshold
        if self.rule:
            self.rule.add_assertion(self)
        return self.rule

    def isLessThan(self, threshold):
        self.operator = "<"
        self.threshold = threshold
        if self.rule:
            self.rule.add_assertion(self)
        return self.rule

    def isGreaterThanOrEqualTo(self, threshold):
        self.operator = ">="
        self.threshold = threshold
        if self.rule:
            self.rule.add_assertion(self)
        return self.rule

    def isLessThanOrEqualTo(self, threshold):
        self.operator = "<="
        self.threshold = threshold
        if self.rule:
            self.rule.add_assertion(self)
        return self.rule

    def isEqualTo(self, threshold):
        self.operator = "=="
        self.threshold = threshold
        if self.rule:
            self.rule.add_assertion(self)
        return self.rule

    def isNotEqualTo(self, threshold):
        self.operator = "!="
        self.threshold = threshold
        if self.rule:
            self.rule.add_assertion(self)
        return self.rule

    def evaluate(self, row):
        """Check if the metric meets the condition.

        Returns False if the metric is missing or NaN to ensure strict filtering.
        """
        if self.metric_name not in row or pd.isna(row[self.metric_name]):
            logger.warning(f"Metric {self.metric_name} missing or NaN for sample {row.get('sample_id', 'unknown')}")
            return False, None

        value = row[self.metric_name]

        # Define comparison functions
        comparisons = {
            ">":  lambda x, y: x > y,
            "<":  lambda x, y: x < y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y
        }

        # Perform the comparison
        passes = comparisons[self.operator](value, self.threshold)
        
        # Format value for display
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        
        # Create a descriptive message
        message = f"{self.metric_name} {self.operator} {self.threshold} (actual: {formatted_value})"
        
        return passes, message


class ConditionGroup:
    """Represents a group of conditions with logical AND or OR."""
    def __init__(self, logic_type="AND"):
        self.logic_type = logic_type
        self.conditions = []

    def add_condition(self, condition):
        """Add a condition (assertion or nested group) to this group."""
        self.conditions.append(condition)
        return self

    def evaluate(self, row):
        """Evaluate all conditions in this group with the specified logic."""
        results = []
        passing = []
        failing = []

        # Collect results from all conditions
        for condition in self.conditions:
            passes, message = condition.evaluate(row)
            results.append(passes)
            
            if not message:
                continue
                
            # Store condition results
            if passes:
                passing.append(message)
            else:
                failing.append(message)

        # Determine if group passes based on logic type
        passes_group = all(results) if self.logic_type == "AND" else any(results)
        
        if passes_group:
            # Group passed - return passing conditions
            if not passing:
                return True, f"{self.logic_type} group matched"
            
            return True, f"{self.logic_type} group matched: {'; '.join(passing)}"
        
        # Group failed - return failing conditions
        if not failing:
            return False, f"{self.logic_type} group failed"
        
        return False, f"{self.logic_type} group failed: {'; '.join(failing)}"


class FilterRule:
    """Defines a filtering rule composed of metric assertions and condition groups."""
    def __init__(self, name, description=None, filter_if_pass=False):
        self.name = name
        self.description = description or name
        self.assertions = []
        self.groups = []
        self.logic_type = "AND"
        self.pending_logic = None
        self.filter_if_pass = filter_if_pass  # Flag to invert rule behavior

    def assertThat(self, metric_name):
        """Start a new metric condition."""
        if self.pending_logic:
            self.logic_type = self.pending_logic
            self.pending_logic = None

        return MetricAssertion(metric_name, self)

    def addGroup(self, group):
        """Add a condition group to this rule."""
        self.groups.append(group)
        return self

    def add_assertion(self, assertion):
        self.assertions.append(assertion)
        return self

    def and_(self):
        self.pending_logic = "AND"
        return self

    def or_(self):
        self.pending_logic = "OR"
        return self

    def evaluate(self, row):
        """Check if the row passes the rule.

        Returns True if no conditions are defined, which might need adjustment based on use case.
        """
        # Early return if no conditions
        if not self.assertions and not self.groups:
            return True, None

        # Process all conditions
        all_results = []
        passing_messages = []
        failing_messages = []
        
        # Evaluate direct assertions
        for assertion in self.assertions:
            passes, message = assertion.evaluate(row)
            all_results.append(passes)
            
            if not message:
                continue
                
            if passes:
                passing_messages.append(message)
            else:
                failing_messages.append(message)
        
        # Evaluate condition groups
        for group in self.groups:
            passes, message = group.evaluate(row)
            all_results.append(passes)
            
            if not message:
                continue
                
            if passes:
                passing_messages.append(message)
            else:
                failing_messages.append(message)
        
        # Determine overall rule result
        rule_passes = all(all_results) if self.logic_type == "AND" else any(all_results)
        
        # Handle normal rules (filter when rule fails)
        if not rule_passes and not self.filter_if_pass:
            if not failing_messages:
                return False, f"Failed {self.name}: Did not meet criteria"
                
            return False, f"Failed {self.name}: {'; '.join(failing_messages)}"
        
        # Handle inverted rules (filter when rule passes)
        if rule_passes and self.filter_if_pass:
            if not passing_messages:
                return False, f"Matches {self.name}: Met problematic criteria"
                
            return False, f"Matches {self.name}: {'; '.join(passing_messages)}"
        
        # Rule passes (no filtering needed)
        return True, None

    @staticmethod
    def _create_condition_from_dict(condition_dict):
        """Create a condition (assertion or group) from a dictionary."""
        # Handle group condition
        if "group" in condition_dict:
            group_dict = condition_dict["group"]
            logic = group_dict.get("logic", "AND")
            group = ConditionGroup(logic)

            for nested_condition in group_dict.get("conditions", []):
                condition = FilterRule._create_condition_from_dict(nested_condition)
                if condition:
                    group.add_condition(condition)

            return group
        
        # Handle regular assertion
        metric = condition_dict.get("metric")
        operator = condition_dict.get("operator")
        threshold = condition_dict.get("threshold")

        if not all([metric, operator, threshold]):
            logger.error(f"Invalid condition: missing required fields")
            return None

        assertion = MetricAssertion(metric)

        if operator == ">":
            assertion.isGreaterThan(threshold)
        elif operator == "<":
            assertion.isLessThan(threshold)
        elif operator == ">=":
            assertion.isGreaterThanOrEqualTo(threshold)
        elif operator == "<=":
            assertion.isLessThanOrEqualTo(threshold)
        elif operator == "==":
            assertion.isEqualTo(threshold)
        elif operator == "!=":
            assertion.isNotEqualTo(threshold)

        return assertion

    @staticmethod
    def from_dict(rule_dict):
        """Create a rule from a dictionary."""
        name = rule_dict.get("name", "unnamed_rule")
        description = rule_dict.get("description", name)
        filter_if_pass = rule_dict.get("filter_if_pass", False)
        
        rule = FilterRule(name, description, filter_if_pass)
        rule.logic_type = rule_dict.get("logic", "AND")
        conditions = rule_dict.get("conditions", [])

        if not isinstance(conditions, list):
            logger.error(f"Invalid conditions for rule {name}: expected list, got {type(conditions)}")
            return rule

        for condition in conditions:
            if "group" in condition:
                group_dict = condition["group"]
                logic = group_dict.get("logic", "AND")
                group = ConditionGroup(logic)

                for nested_condition in group_dict.get("conditions", []):
                    condition_obj = FilterRule._create_condition_from_dict(nested_condition)
                    if condition_obj:
                        group.add_condition(condition_obj)

                rule.addGroup(group)
                continue

            # Regular condition
            metric = condition.get("metric")
            operator = condition.get("operator")
            threshold = condition.get("threshold")

            if not all([metric, operator, threshold]):
                logger.error(f"Invalid condition in rule {name}: missing required fields")
                continue

            assertion = rule.assertThat(metric)

            if operator == ">":
                assertion.isGreaterThan(threshold)
            elif operator == "<":
                assertion.isLessThan(threshold)
            elif operator == ">=":
                assertion.isGreaterThanOrEqualTo(threshold)
            elif operator == "<=":
                assertion.isLessThanOrEqualTo(threshold)
            elif operator == "==":
                assertion.isEqualTo(threshold)
            elif operator == "!=":
                assertion.isNotEqualTo(threshold)

        return rule


class RuleSet:
    """Manages a collection of filtering rules."""
    @staticmethod
    def get_default_rules():
        """Provide default rules based on metric thresholds."""
        rules = []

        # Logical consistency rule
        summac_rule = FilterRule("logical_consistency", "Checks logical consistency")
        summac_rule.assertThat("summac").isGreaterThanOrEqualTo(0.0)
        rules.append(summac_rule)

        # Semantic similarity rule
        bertscore_rule = FilterRule("semantic_similarity", "Checks semantic similarity")
        bertscore_rule.assertThat("bertscore-f1").isGreaterThanOrEqualTo(0.75)
        rules.append(bertscore_rule)

        # Factual correctness rule
        factcc_rule = FilterRule("factual_correctness", "Checks factual correctness")
        factcc_rule.assertThat("factcc").isGreaterThanOrEqualTo(0.5)
        rules.append(factcc_rule)

        # Content alignment rule
        alignscore_rule = FilterRule("content_alignment", "Checks content alignment")
        alignscore_rule.assertThat("alignscore").isGreaterThanOrEqualTo(0.15)
        rules.append(alignscore_rule)

        # Text quality rule
        bleurt_rule = FilterRule("quality_check", "Checks text quality")
        bleurt_rule.assertThat("bleurt").isGreaterThanOrEqualTo(0.15)
        rules.append(bleurt_rule)

        return rules

    @staticmethod
    def load_from_file(filepath):
        """Load rules from a JSON file."""
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"Rules file not found: {filepath}")
            return None

        rules_data = load_json(filepath)
        rules = [FilterRule.from_dict(rule_data) for rule_data in rules_data]

        logger.info(f"Loaded {len(rules)} rules from {filepath}")
        return rules


class ContentFilter:
    """Filters out text samples with problematic content based on predefined patterns."""
    def __init__(self):
        self.mismatch_pattern = re.compile('|'.join(re.escape(p) for p in ContentPatterns.MISMATCH), re.IGNORECASE)
        self.placeholder_pattern = re.compile('|'.join(ContentPatterns.PLACEHOLDER), re.IGNORECASE)
        # Track detailed removal info
        self.removed_originals = {}  # {sample_id: info}
        self.removed_variants = {}   # {sample_id: [variant_info, ...]}

    def filter(self, df):
        """Filter out problematic content from the dataframe."""
        original_count = len(df)

        # Separate original answers and variants
        original_answers = df[df["text_type"] == "original"]
        variants = df[df["text_type"] == "variant"]

        # Find problematic originals
        original_mask = self._get_problematic_mask(original_answers["text"])
        problematic_originals = original_answers[original_mask]
        
        # Store problematic original sample IDs with their datasets
        for _, row in problematic_originals.iterrows():
            sample_id = row["sample_id"]
            dataset = row.get("dataset", "unknown")
            question = row.get("question", "")
            text = row.get("text", "")  # Preserve the problematic text
            
            self.removed_originals[sample_id] = {
                "dataset": dataset,
                "question": question,
                "reason": "content_issue",
                "description": "Problematic content in original text",
                "original_text": text  # Include the original text in the report
            }
        
        problematic_sample_ids = set(problematic_originals["sample_id"])

        # Remove samples with problematic originals
        clean_df = df[~df["sample_id"].isin(problematic_sample_ids)]

        # Find problematic variants in remaining data
        remaining_variants = clean_df[clean_df["text_type"] == "variant"]
        variant_mask = self._get_problematic_mask(remaining_variants["text"])
        problematic_variants = remaining_variants[variant_mask]
        
        # Get sample_ids of problematic variants
        problematic_variant_sample_ids = set(problematic_variants["sample_id"])
        
        # Store problematic variant info
        for _, row in problematic_variants.iterrows():
            sample_id = row["sample_id"]
            dataset = row.get("dataset", "unknown")
            complexity = row.get("complexity_level")
            model = row.get("model_used", "unknown")
            temperature = row.get("temperature", "unknown")
            question = row.get("question", "")
            text = row.get("text", "")  # Preserve the problematic text
            
            variant_info = {
                "dataset": dataset,
                "complexity_level": complexity,
                "model": model,
                "temperature": temperature,
                "question": question,
                "reason": "content_issue",
                "description": "Problematic content in variant text",
                "variant_text": text  # Include the variant text in the report
            }
            
            if sample_id not in self.removed_variants:
                self.removed_variants[sample_id] = []
            
            self.removed_variants[sample_id].append(variant_info)

        # Remove all rows with problematic variant sample_ids
        clean_df = clean_df[~clean_df["sample_id"].isin(problematic_variant_sample_ids)]

        # Calculate statistics
        stats = {
            "original_count": original_count,
            "problematic_originals": len(problematic_originals),
            "problematic_variants": len(problematic_variants),
            "remaining_count": len(clean_df)
        }

        logger.info(f"Removed {stats['problematic_originals']} samples and {stats['problematic_variants']} variants")
        return clean_df, stats

    def _get_problematic_mask(self, texts):
        """Create a mask for problematic texts."""
        if texts.empty:
            return pd.Series(False, index=texts.index)

        mismatch_mask = texts.str.contains(self.mismatch_pattern, na=False)
        placeholder_mask = texts.str.contains(self.placeholder_pattern, regex=True, na=False)

        return mismatch_mask | placeholder_mask


class MetricFilter:
    """Filters out samples based on metric rules applied to score data."""
    def __init__(self, rules):
        self.rules = rules
        # Track detailed removal info by sample_id
        self.removed_by_rules = {}  # {sample_id: [failure_info, ...]}

    def filter(self, samples_df, scores_df):
        """Filter samples using metric rules."""
        if scores_df.empty:
            logger.info("No scores provided, skipping metric filtering")
            return samples_df, {"rule_filtering_skipped": True}

        original_count = len(samples_df)

        # Create a lookup dictionary for text_type from samples_df
        text_type_lookup = {}
        for _, row in samples_df.iterrows():
            if "sample_id" in row and "text_type" in row:
                sample_id = row["sample_id"]
                text_type = row["text_type"]
                text_type_lookup[sample_id] = text_type

        # Identify failed rows
        failed = self._evaluate_rules(scores_df)
        failed_indices = list(failed.keys())

        # Store detailed failure information
        for idx, reasons in failed.items():
            if idx not in scores_df.index:
                continue
            
            row = scores_df.loc[idx]
            sample_id = row.get("sample_id", f"unknown_at_{idx}")
            
            # Skip if we don't have a sample_id
            if sample_id == f"unknown_at_{idx}":
                continue
                
            # Get row details
            dataset = row.get("dataset", "unknown")
            text_type = text_type_lookup.get(sample_id, "variant")  # Default to variant if not found
            complexity_level = row.get("complexity_level")
            model = row.get("model", row.get("model_used", "unknown"))
            temperature = row.get("temperature", "unknown")
            question = row.get("question", "")
            original_text = row.get("original_text", "")
            variant_text = row.get("variant_text", "")
            
            # Extract main failure reason
            main_reason = reasons[0].split(":")[0] if reasons else "unknown_rule_failure"
            
            # Store info about this failure
            if sample_id not in self.removed_by_rules:
                self.removed_by_rules[sample_id] = []
                
            failure_info = {
                "dataset": dataset,
                "text_type": text_type,
                "complexity_level": complexity_level,
                "model": model,
                "temperature": temperature,
                "question": question,
                "original_text": original_text,
                "variant_text": variant_text,
                "reason": main_reason,
                "description": reasons[0] if reasons else "Unknown failure",
                "all_failures": reasons
            }
            
            self.removed_by_rules[sample_id].append(failure_info)

        # Remove failed rows
        filtered_df = samples_df[~samples_df.index.isin(failed_indices)]

        # Calculate statistics
        stats = {
            "original_count": original_count,
            "filtered_count": len(failed_indices),
            "remaining_count": len(filtered_df)
        }

        logger.info(f"Removed {stats['filtered_count']} rows due to metric rules")
        return filtered_df, stats

    def _evaluate_rules(self, scores_df):
        """Check which rows fail the rules."""
        failed = {}

        for idx, row in scores_df.iterrows():
            failures = []

            for rule in self.rules:
                passes, failure_reason = rule.evaluate(row)

                if not passes:
                    failures.append(failure_reason)

            if failures:
                failed[idx] = failures

        return failed


def parse_exclusions(exclude_ids):
    """Parse exclusion IDs for samples or variants."""
    if not exclude_ids:
        return [], {}
        
    full_exclusions = []
    variant_exclusions = {}

    for entry in exclude_ids:
        parts = entry.split(":")
        if len(parts) not in [1, 2, 3]:
            logger.warning(f"Invalid exclusion format: {entry}")
            continue

        # Handle dataset:sample_id format
        if len(parts) == 2:
            dataset, sample_id = parts
            full_id = f"{dataset}:{sample_id}"
            full_exclusions.append(full_id)
            logger.info(f"Excluding sample {full_id}")
            continue
            
        # Handle just sample_id format (legacy)
        if len(parts) == 1:
            sample_id = parts[0]
            full_exclusions.append(sample_id)
            logger.info(f"Excluding sample ID {sample_id}")
            continue

        # Handle dataset:sample_id:gen_idx,var_idx format
        dataset, sample_id, indices = parts
        full_id = f"{dataset}:{sample_id}"
        
        if "," not in indices:
            logger.warning(f"Invalid variant indices in {entry}, should be gen_idx,var_idx")
            continue
            
        gen_idx, var_idx = indices.split(",")
        if not gen_idx.isdigit() or not var_idx.isdigit():
            logger.warning(f"Invalid variant indices in {entry}, should be numeric")
            continue

        gen_idx, var_idx = int(gen_idx), int(var_idx)
        if gen_idx < 0 or var_idx < 0:
            logger.warning(f"Negative indices in {entry}")
            continue

        variant_exclusions.setdefault(full_id, []).append((gen_idx, var_idx))
        logger.info(f"Excluding variant ({gen_idx},{var_idx}) from {full_id}")

    return full_exclusions, variant_exclusions


class Phase4Pipeline:
    """Orchestrates the filtering pipeline for text samples based on content and metric rules."""
    def __init__(self, metrics_file, scores_file=None, output_dir=None, exclude_ids=None, rules_file=None):
        self.metrics_file = Path(metrics_file)
        self.scores_file = Path(scores_file) if scores_file else None
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/phase4")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exclude_ids, self.exclude_variants = parse_exclusions(exclude_ids or [])

        # Load rules
        rules_path = Path(rules_file) if rules_file else None
        self.rules = RuleSet.load_from_file(rules_path) if rules_path and rules_path.exists() else RuleSet.get_default_rules()

        # Initialize filters
        self.content_filter = ContentFilter()
        self.metric_filter = MetricFilter(self.rules)

        # Data storage
        self.metrics_df = pd.DataFrame()
        self.scores_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.stats = {}
        
        # Tracking manually excluded samples
        self.manually_excluded = {}
        
        # Track complexity level statistics
        self.complexity_stats = defaultdict(Counter)

    def run(self):
        """Execute the filtering pipeline."""
        logger.info(f"Starting pipeline with {self.metrics_file}")

        self._load_data()

        if self.metrics_df.empty:
            logger.error("No data to process")
            return pd.DataFrame()

        self._apply_filters()
        self._save_results()

        logger.info(f"Pipeline complete: {len(self.filtered_df)} rows remaining")
        return self.filtered_df

    def _load_data(self):
        """Load data from files."""
        if not self.metrics_file.exists():
            logger.error(f"Metrics file missing: {self.metrics_file}")
            return

        self.metrics_df = pd.read_csv(self.metrics_file)
        logger.info(f"Loaded {len(self.metrics_df)} rows from {self.metrics_file}")

        if not self.scores_file or not self.scores_file.exists():
            logger.warning("No scores file, metric filtering will be skipped")
            return

        self.scores_df = pd.read_csv(self.scores_file)
        logger.info(f"Loaded {len(self.scores_df)} rows from {self.scores_file}")

    def _apply_filters(self):
        """Apply all filtering steps."""
        df = self.metrics_df.copy()
        self.stats["initial_count"] = len(df)
        
        # Calculate initial complexity distribution
        self._update_complexity_stats(df, "initial")

        # Exclude specified IDs
        if self.exclude_ids:
            # Track samples being manually excluded
            for sample_id in self.exclude_ids:
                rows = df[df["sample_id"] == sample_id]
                if not rows.empty:
                    first_row = rows.iloc[0]
                    dataset = first_row.get("dataset", "unknown")
                    question = first_row.get("question", "")
                    
                    self.manually_excluded[sample_id] = {
                        "dataset": dataset,
                        "question": question,
                        "reason": "manual_exclusion",
                        "description": "Manually excluded"
                    }
                    
                    # Track complexity levels of manual exclusions
                    for _, row in rows.iterrows():
                        text_type = row.get("text_type", "unknown")
                        complexity = row.get("complexity_level")
                        if text_type == "variant" and complexity is not None:
                            self.complexity_stats["manual_exclusion"][complexity] += 1
            
            df = df[~df["sample_id"].isin(self.exclude_ids)]
            self.stats["manual_exclusions"] = self.stats["initial_count"] - len(df)
            logger.info(f"Manually excluded {self.stats['manual_exclusions']} rows")

        # Filter content issues
        df, content_stats = self.content_filter.filter(df)
        self.stats["content_filtering"] = content_stats
        
        # Track complexity levels for content filtering
        for sample_id, variants in self.content_filter.removed_variants.items():
            for variant in variants:
                complexity = variant.get("complexity_level")
                if complexity is not None:
                    self.complexity_stats["content_issue"][complexity] += 1

        # Filter based on metrics
        if not self.scores_df.empty:
            df, rule_stats = self.metric_filter.filter(df, self.scores_df)
            self.stats["rule_filtering"] = rule_stats
            
            # Track complexity levels for rule filtering
            for sample_id, failures in self.metric_filter.removed_by_rules.items():
                for failure in failures:
                    complexity = failure.get("complexity_level")
                    if complexity is not None:
                        reason = failure.get("reason", "unknown_rule")
                        self.complexity_stats[reason][complexity] += 1

        # Final statistics
        self.stats["final_count"] = len(df)
        self.stats["total_filtered"] = self.stats["initial_count"] - len(df)
        self.stats["percentage_filtered"] = (self.stats["total_filtered"] / self.stats["initial_count"]) * 100 if self.stats["initial_count"] else 0
        
        # Calculate final complexity distribution
        self._update_complexity_stats(df, "final")

        self.filtered_df = df

    def _update_complexity_stats(self, df, stage):
        """Update complexity level statistics for a given dataframe stage."""
        variants = df[df["text_type"] == "variant"]
        if "complexity_level" in variants.columns:
            complexity_counts = variants["complexity_level"].value_counts().to_dict()
            self.complexity_stats[f"{stage}_distribution"] = Counter(complexity_counts)

    def _save_results(self):
        """Save the filtered data and stats."""
        # Save filtered data
        filtered_path = self.output_dir / "filtered_df.csv"
        self.filtered_df.to_csv(filtered_path, index=False)
        logger.info(f"Saved {len(self.filtered_df)} rows to {filtered_path}")

        # Save a sample if large
        if len(self.filtered_df) > 100:
            sample_path = self.output_dir / "filtered_df_sample.csv"
            self.filtered_df.head(100).to_csv(sample_path, index=False)
            logger.info(f"Saved sample of 100 rows to {sample_path}")

        # Prepare detailed removal information
        kept_samples = set(self.filtered_df["sample_id"].unique())
        all_samples = set(self.metrics_df["sample_id"].unique())
        removed_samples = all_samples - kept_samples
        
        # Organize datasets and their samples
        datasets = defaultdict(lambda: {"samples": {}})
        
        # Generate dataset-level stats
        dataset_counts = Counter()
        reason_counts = Counter()
        complexity_reason_counts = defaultdict(Counter)
        
        # Add manually excluded samples
        for sample_id, info in self.manually_excluded.items():
            dataset = info["dataset"]
            question = info.get("question", "")
            reason = info["reason"]
            
            dataset_counts[dataset] += 1
            reason_counts[reason] += 1
            
            # Try to find the original text from the metrics dataframe
            original_text = ""
            original_rows = self.metrics_df[(self.metrics_df["sample_id"] == sample_id) & 
                                          (self.metrics_df["text_type"] == "original")]
            if not original_rows.empty:
                original_text = original_rows.iloc[0].get("text", "")
            
            sample_info = {
                "reason": reason,
                "description": info["description"],
                "text_type": "all",
                "question": question
            }
            
            # Only add text if it has content
            if original_text:
                sample_info["original_text"] = original_text
                
            datasets[dataset]["samples"][sample_id] = sample_info
        
        # Add content filtered originals
        for sample_id, info in self.content_filter.removed_originals.items():
            dataset = info["dataset"]
            question = info.get("question", "")
            reason = info["reason"]
            original_text = info.get("original_text", "")
            
            dataset_counts[dataset] += 1
            reason_counts[reason] += 1
            
            sample_info = {
                "reason": reason,
                "description": info.get("description", "Problematic content in original text"),
                "text_type": "original",
                "question": question
            }
            
            # Only add text if it has content
            if original_text:
                sample_info["original_text"] = original_text
                
            datasets[dataset]["samples"][sample_id] = sample_info
        
        # Add rule-filtered samples
        for sample_id, failures in self.metric_filter.removed_by_rules.items():
            # Skip if already excluded for another reason
            already_added = False
            for dataset_info in datasets.values():
                if sample_id in dataset_info["samples"]:
                    already_added = True
                    break
            
            if already_added:
                continue
                
            # Get info from first failure
            if not failures:
                continue
                
            first_failure = failures[0]
            dataset = first_failure["dataset"]
            text_type = first_failure["text_type"]
            complexity = first_failure.get("complexity_level")
            reason = first_failure["reason"]
            description = first_failure["description"]
            question = first_failure.get("question", "")
            original_text = first_failure.get("original_text", "")
            variant_text = first_failure.get("variant_text", "")
            
            # Add to counters
            dataset_counts[dataset] += 1
            reason_counts[reason] += 1
            
            if complexity is not None:
                complexity_reason_counts[reason][complexity] += 1
            
            # Add to datasets structure
            sample_info = {
                "reason": reason,
                "description": description,
                "text_type": text_type,
                "complexity_level": complexity,
                "question": question
            }
            
            # Only add text fields if they have content
            if original_text:
                sample_info["original_text"] = original_text
            if variant_text:
                sample_info["variant_text"] = variant_text
                
            datasets[dataset]["samples"][sample_id] = sample_info
        
        # Create comprehensive filtering report
        filtering_report = {
            "summary": {
                "initial_count": self.stats["initial_count"],
                "final_count": self.stats["final_count"],
                "total_filtered": self.stats["total_filtered"],
                "percentage_filtered": self.stats["percentage_filtered"],
                "removal_counts_by_reason": dict(reason_counts),
                "removal_counts_by_dataset": dict(dataset_counts),
                "complexity_distribution": {
                    k: dict(v) for k, v in self.complexity_stats.items()
                },
                "complexity_removal_by_reason": {
                    reason: dict(counts) for reason, counts in complexity_reason_counts.items()
                },
                "total_removed_samples": len(removed_samples)
            },
            "datasets": dict(datasets)
        }
        
        # Save the comprehensive report
        report_path = self.output_dir / "filtering_report.json"
        save_json(filtering_report, report_path)
        logger.info(f"Saved comprehensive filtering report to {report_path}")
        
        # Save applied rules separately
        rules_info = [
            {
                "name": rule.name,
                "description": rule.description,
                "logic": rule.logic_type,
                "filter_if_pass": rule.filter_if_pass,
                "conditions": self._serialize_rule_conditions(rule)
            }
            for rule in self.rules
        ]
        
        rules_path = self.output_dir / "applied_rules.json"
        save_json(rules_info, rules_path)
        logger.info(f"Saved {len(rules_info)} rules to {rules_path}")
        
    def _serialize_rule_conditions(self, rule):
        """Serialize a rule's conditions including nested groups."""
        conditions = []
        
        # Add direct assertions
        for assertion in rule.assertions:
            conditions.append({
                "metric": assertion.metric_name,
                "operator": assertion.operator,
                "threshold": assertion.threshold
            })
        
        # Add groups
        for group in rule.groups:
            group_conditions = []
            
            for condition in group.conditions:
                if isinstance(condition, MetricAssertion):
                    group_conditions.append({
                        "metric": condition.metric_name,
                        "operator": condition.operator,
                        "threshold": condition.threshold
                    })
                elif isinstance(condition, ConditionGroup):
                    # Handle nested groups recursively (not implemented for simplicity)
                    pass
            
            conditions.append({
                "group": {
                    "logic": group.logic_type,
                    "conditions": group_conditions
                }
            })
            
        return conditions


def main():
    """Run the pipeline from the command line."""   
    parser = argparse.ArgumentParser(description="Phase 4: Filter low-quality text samples")

    parser.add_argument("--metrics-file", required=True, help="Path to metrics CSV")
    parser.add_argument("--scores-file", help="Path to scores CSV (optional)")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--exclude-ids", nargs="+", help="Sample IDs to exclude (e.g., 'dataset:id' or 'dataset:id:gen_idx,var_idx' or just 'id')")
    parser.add_argument("--rules-file", help="Path to custom rules JSON")

    args = parser.parse_args()

    pipeline = Phase4Pipeline(
        metrics_file=args.metrics_file,
        scores_file=args.scores_file,
        output_dir=args.output_dir,
        exclude_ids=args.exclude_ids,
        rules_file=args.rules_file
    )

    pipeline.run()


if __name__ == "__main__":
    main()