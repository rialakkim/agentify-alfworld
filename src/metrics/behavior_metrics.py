"""Behavior metrics for evaluating white agent performance.

This module provides metrics to analyze:
1. Repeated steps (consecutive identical actions)
2. Cleanup behavior (did agent close what it opened)
3. Action cycles (repetitive patterns)
4. Overall behavioral score
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ActionTracker:
    """Tracks actions taken during an episode for metric calculation."""
    
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    
    # Track open/close state
    opened_items: Dict[str, bool] = field(default_factory=dict)
    
    def add_action(self, action: str, observation: str = ""):
        """Record an action and its resulting observation."""
        self.actions.append(action.lower().strip())
        self.observations.append(observation)
        self._update_open_close_state(action)
    
    def _update_open_close_state(self, action: str):
        """Track items that have been opened/closed."""
        action_lower = action.lower().strip()
        
        # Match "open X" or "close X" patterns
        open_match = re.match(r'^open\s+(.+)$', action_lower)
        close_match = re.match(r'^close\s+(.+)$', action_lower)
        
        if open_match:
            item = open_match.group(1)
            self.opened_items[item] = True  # True = currently open
        elif close_match:
            item = close_match.group(1)
            if item in self.opened_items:
                self.opened_items[item] = False  # False = closed
    
    def get_unclosed_items(self) -> List[str]:
        """Return list of items that were opened but never closed."""
        return [item for item, is_open in self.opened_items.items() if is_open]
    
    def clear(self):
        """Reset the tracker for a new episode."""
        self.actions.clear()
        self.observations.clear()
        self.opened_items.clear()


class BehaviorMetrics:
    """Calculate behavioral metrics for white agent evaluation."""
    
    # Actions that involve opening something
    OPEN_ACTIONS = {'open'}
    # Actions that involve closing something
    CLOSE_ACTIONS = {'close'}
    # Navigation actions for cycle detection
    NAVIGATION_ACTIONS = {'go to', 'go'}
    
    @staticmethod
    def calculate_repeated_steps(actions: List[str]) -> Dict[str, Any]:
        """
        Calculate how often the white agent repeated consecutive steps.
        
        Ex: "pick up spoon", "pick up spoon", "pick up spoon"
        
        Returns:
            Dict with:
            - repeated_count: number of times actions were repeated consecutively
            - repeated_sequences: list of (action, count) tuples for repeated actions
            - repetition_rate: percentage of actions that were repetitions
        """
        if len(actions) < 2:
            return {
                "repeated_count": 0,
                "repeated_sequences": [],
                "repetition_rate": 0.0,
                "max_repeat_length": 0,
            }
        
        repeated_count = 0
        repeated_sequences = []
        current_action = actions[0]
        current_count = 1
        
        for i in range(1, len(actions)):
            if actions[i] == current_action:
                current_count += 1
            else:
                if current_count > 1:
                    repeated_sequences.append((current_action, current_count))
                    repeated_count += current_count - 1  # Count extra repetitions
                current_action = actions[i]
                current_count = 1
        
        # Don't forget the last sequence
        if current_count > 1:
            repeated_sequences.append((current_action, current_count))
            repeated_count += current_count - 1
        
        max_repeat = max([count for _, count in repeated_sequences], default=0)
        
        return {
            "repeated_count": repeated_count,
            "repeated_sequences": repeated_sequences,
            "repetition_rate": repeated_count / len(actions) if actions else 0.0,
            "max_repeat_length": max_repeat,
        }
    
    @staticmethod
    def calculate_cleanup_metric(tracker: ActionTracker) -> Dict[str, Any]:
        """
        Calculate cleanup metric - how often the agent left things open.
        
        Measures if the agent closes drawers, cabinets, etc. after opening them.
        
        Returns:
            Dict with:
            - items_opened: total items opened
            - items_closed: total items that were closed after opening
            - unclosed_items: list of items left open
            - cleanup_rate: percentage of opened items that were closed
        """
        items_opened = len(tracker.opened_items)
        unclosed_items = tracker.get_unclosed_items()
        items_closed = items_opened - len(unclosed_items)
        
        cleanup_rate = items_closed / items_opened if items_opened > 0 else 1.0
        
        return {
            "items_opened": items_opened,
            "items_closed": items_closed,
            "unclosed_items": unclosed_items,
            "cleanup_rate": cleanup_rate,
        }
    
    @staticmethod
    def calculate_cycle_metric(actions: List[str], min_cycle_length: int = 2, max_cycle_length: int = 5) -> Dict[str, Any]:
        """
        Calculate cycle metric - detect repetitive navigation patterns.
        
        Ex: "go to couch", "go to table", "go to couch", "go to table"
        
        Args:
            actions: List of actions taken
            min_cycle_length: Minimum length of a cycle pattern (default: 2)
            max_cycle_length: Maximum length of a cycle pattern to detect (default: 5)
        
        Returns:
            Dict with:
            - cycles_detected: number of cycles found
            - cycle_patterns: list of detected cycle patterns
            - cycle_rate: percentage of actions that were part of cycles
            - wasted_steps: estimated steps wasted in cycles
        """
        if len(actions) < min_cycle_length * 2:
            return {
                "cycles_detected": 0,
                "cycle_patterns": [],
                "cycle_rate": 0.0,
                "wasted_steps": 0,
            }
        
        cycles_detected = []
        actions_in_cycles = set()
        
        # Check for cycles of various lengths
        for cycle_len in range(min_cycle_length, max_cycle_length + 1):
            i = 0
            while i <= len(actions) - cycle_len * 2:
                # Get potential cycle pattern
                pattern = tuple(actions[i:i + cycle_len])
                
                # Count how many times this pattern repeats consecutively
                repeat_count = 1
                j = i + cycle_len
                while j + cycle_len <= len(actions):
                    if tuple(actions[j:j + cycle_len]) == pattern:
                        repeat_count += 1
                        j += cycle_len
                    else:
                        break
                
                if repeat_count >= 2:
                    cycles_detected.append({
                        "pattern": list(pattern),
                        "length": cycle_len,
                        "repetitions": repeat_count,
                        "start_index": i,
                    })
                    # Mark these actions as part of a cycle
                    for idx in range(i, j):
                        actions_in_cycles.add(idx)
                    i = j  # Skip past this cycle
                else:
                    i += 1
        
        # Calculate wasted steps (steps after first cycle occurrence)
        wasted_steps = 0
        for cycle in cycles_detected:
            # All repetitions after the first are "wasted"
            wasted_steps += cycle["length"] * (cycle["repetitions"] - 1)
        
        return {
            "cycles_detected": len(cycles_detected),
            "cycle_patterns": cycles_detected,
            "cycle_rate": len(actions_in_cycles) / len(actions) if actions else 0.0,
            "wasted_steps": wasted_steps,
        }
    
    @staticmethod
    def calculate_exploration_efficiency(actions: List[str]) -> Dict[str, Any]:
        """
        Calculate how efficiently the agent explored the environment.
        
        Tracks unique locations visited vs total navigation actions.
        """
        navigation_actions = []
        visited_locations = set()
        
        for action in actions:
            action_lower = action.lower().strip()
            # Check for "go to X" pattern
            match = re.match(r'^go\s+to\s+(.+)$', action_lower)
            if match:
                location = match.group(1)
                navigation_actions.append(location)
                visited_locations.add(location)
        
        unique_visits = len(visited_locations)
        total_nav = len(navigation_actions)
        
        return {
            "unique_locations_visited": unique_visits,
            "total_navigation_actions": total_nav,
            "exploration_efficiency": unique_visits / total_nav if total_nav > 0 else 1.0,
            "revisit_count": total_nav - unique_visits,
        }
    
    @staticmethod
    def calculate_overall_score(
        repeated_metrics: Dict[str, Any],
        cleanup_metrics: Dict[str, Any],
        cycle_metrics: Dict[str, Any],
        task_completed: bool,
        total_steps: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Calculate an overall behavioral score combining all metrics.
        
        Score components:
        - Task completion: 40% weight
        - Efficiency (steps used): 20% weight
        - No repetitions: 15% weight
        - Cleanup behavior: 15% weight
        - No cycles: 10% weight
        
        Returns:
            Dict with overall score (0-100) and breakdown
        """
        # Task completion score (0 or 40)
        completion_score = 40.0 if task_completed else 0.0
        
        # Efficiency score (0-20) - fewer steps is better
        if task_completed and max_steps > 0:
            efficiency_ratio = 1.0 - (total_steps / max_steps)
            efficiency_score = max(0, min(20, efficiency_ratio * 20 + 10))  # 10-20 for completing
        else:
            efficiency_score = 0.0
        
        # Repetition score (0-15) - fewer repetitions is better
        repetition_rate = repeated_metrics.get("repetition_rate", 0.0)
        repetition_score = max(0, 15 * (1.0 - repetition_rate))
        
        # Cleanup score (0-15) - higher cleanup rate is better
        cleanup_rate = cleanup_metrics.get("cleanup_rate", 1.0)
        cleanup_score = 15 * cleanup_rate
        
        # Cycle score (0-10) - fewer cycles is better
        cycle_rate = cycle_metrics.get("cycle_rate", 0.0)
        cycle_score = max(0, 10 * (1.0 - cycle_rate))
        
        overall_score = completion_score + efficiency_score + repetition_score + cleanup_score + cycle_score
        
        return {
            "overall_score": round(overall_score, 2),
            "max_possible_score": 100,
            "breakdown": {
                "completion_score": round(completion_score, 2),
                "efficiency_score": round(efficiency_score, 2),
                "repetition_score": round(repetition_score, 2),
                "cleanup_score": round(cleanup_score, 2),
                "cycle_score": round(cycle_score, 2),
            },
            "grade": BehaviorMetrics._score_to_grade(overall_score),
        }
    
    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    @staticmethod
    def compute_all_metrics(
        tracker: ActionTracker,
        task_completed: bool,
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Compute all behavior metrics for an episode.
        
        Args:
            tracker: ActionTracker with recorded actions
            task_completed: Whether the task was successfully completed
            max_steps: Maximum allowed steps
        
        Returns:
            Dict containing all metrics
        """
        actions = tracker.actions
        total_steps = len(actions)
        
        repeated_metrics = BehaviorMetrics.calculate_repeated_steps(actions)
        cleanup_metrics = BehaviorMetrics.calculate_cleanup_metric(tracker)
        cycle_metrics = BehaviorMetrics.calculate_cycle_metric(actions)
        exploration_metrics = BehaviorMetrics.calculate_exploration_efficiency(actions)
        
        overall = BehaviorMetrics.calculate_overall_score(
            repeated_metrics,
            cleanup_metrics,
            cycle_metrics,
            task_completed,
            total_steps,
            max_steps,
        )
        
        return {
            "summary": {
                "task_completed": task_completed,
                "total_steps": total_steps,
                "overall_score": overall["overall_score"],
                "grade": overall["grade"],
            },
            "repeated_steps": repeated_metrics,
            "cleanup": cleanup_metrics,
            "cycles": cycle_metrics,
            "exploration": exploration_metrics,
            "overall": overall,
            "action_history": actions,
        }
