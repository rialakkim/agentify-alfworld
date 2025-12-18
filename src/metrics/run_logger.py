"""Run logger for saving detailed evaluation logs.

Saves evaluation results to timestamped log files for analysis.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


class RunLogger:
    """Logger for saving detailed run information to files."""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the run logger.
        
        Args:
            log_dir: Directory to save logs. Defaults to 'logs' in project root.
        """
        if log_dir is None:
            # Default to logs directory in project root
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.game_logs: List[Dict[str, Any]] = []
        self.run_metadata: Dict[str, Any] = {}
    
    def set_run_metadata(self, metadata: Dict[str, Any]):
        """Set metadata for this evaluation run."""
        self.run_metadata = {
            "timestamp": self.run_timestamp,
            "start_time": datetime.now().isoformat(),
            **metadata
        }
    
    def log_game(self, game_idx: int, game_result: Dict[str, Any]):
        """
        Log results from a single game/episode.
        
        Args:
            game_idx: Index of the game
            game_result: Dict containing game results and metrics
        """
        log_entry = {
            "game_idx": game_idx,
            "timestamp": datetime.now().isoformat(),
            **game_result
        }
        self.game_logs.append(log_entry)
        
        # Save individual game log
        game_file = self.run_dir / f"game_{game_idx:03d}.json"
        with open(game_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
    
    def log_action(self, game_idx: int, step: int, action: str, observation: str, 
                   admissible_commands: List[str] = None):
        """
        Log an individual action (for detailed step-by-step logs).
        
        Args:
            game_idx: Index of the game
            step: Step number
            action: Action taken by white agent
            observation: Observation received after action
            admissible_commands: List of available commands
        """
        actions_file = self.run_dir / f"game_{game_idx:03d}_actions.jsonl"
        action_entry = {
            "step": step,
            "action": action,
            "observation_preview": observation[:200] if observation else "",
            "admissible_count": len(admissible_commands) if admissible_commands else 0,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(actions_file, 'a') as f:
            f.write(json.dumps(action_entry) + "\n")
    
    def finalize_run(self, aggregate_metrics: Dict[str, Any]) -> str:
        """
        Finalize the run and save summary.
        
        Args:
            aggregate_metrics: Aggregated metrics across all games
            
        Returns:
            Path to the summary file
        """
        summary = {
            "run_metadata": self.run_metadata,
            "end_time": datetime.now().isoformat(),
            "aggregate_metrics": aggregate_metrics,
            "total_games": len(self.game_logs),
            "game_summaries": [
                {
                    "game_idx": g["game_idx"],
                    "won": g.get("won", False),
                    "steps": g.get("steps", 0),
                    "score": g.get("score", 0),
                    "behavior_score": g.get("behavior_metrics", {}).get("summary", {}).get("overall_score", 0),
                }
                for g in self.game_logs
            ]
        }
        
        # Save summary
        summary_file = self.run_dir / "run_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed metrics report
        report = self._generate_report(aggregate_metrics)
        report_file = self.run_dir / "run_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nLogs saved to: {self.run_dir}")
        return str(summary_file)
    
    def _generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 70,
            "ALFWORLD EVALUATION REPORT",
            f"Run ID: {self.run_timestamp}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Games: {metrics.get('total_games', 0)}",
            f"Wins: {metrics.get('wins', 0)}",
            f"Success Rate: {metrics.get('success_rate', 0):.1%}",
            f"Average Steps: {metrics.get('avg_steps', 0):.1f}",
            f"Time Used: {metrics.get('time_used', 0):.2f}s",
            "",
            "BEHAVIOR METRICS (Averaged)",
            "-" * 40,
        ]
        
        behavior = metrics.get("behavior_metrics", {})
        
        # Repeated steps
        repeated = behavior.get("repeated_steps", {})
        lines.extend([
            "",
            "Repeated Steps:",
            f"   Total repetitions: {repeated.get('avg_repeated_count', 0):.1f}",
            f"   Repetition rate: {repeated.get('avg_repetition_rate', 0):.1%}",
            f"   Max consecutive repeats: {repeated.get('max_repeat_length', 0)}",
        ])
        
        # Cleanup
        cleanup = behavior.get("cleanup", {})
        lines.extend([
            "",
            "Cleanup Behavior:",
            f"   Total opens: {cleanup.get('avg_total_opens', 0):.1f}",
            f"   Correctly closed: {cleanup.get('avg_correct_close_count', 0):.1f}",
            f"   Remaining open: {cleanup.get('avg_remaining_open_count', 0):.1f}",
            f"   Cleanup rate: {cleanup.get('avg_cleanup_rate', 0):.1%}",
        ])
        
        # Cycles
        cycles = behavior.get("cycles", {})
        lines.extend([
            "",
            "Action Cycles:",
            f"   Cycles detected: {cycles.get('avg_cycles_detected', 0):.1f}",
            f"   Cycle rate: {cycles.get('avg_cycle_rate', 0):.1%}",
            f"   Wasted steps: {cycles.get('avg_wasted_steps', 0):.1f}",
        ])
        
        # Overall
        overall = behavior.get("overall", {})
        lines.extend([
            "",
            "Overall Score:",
            f"   Average Score: {overall.get('avg_score', 0):.1f}/100",
            f"   Grade: {overall.get('avg_grade', 'N/A')}",
        ])
        
        if overall.get("breakdown"):
            breakdown = overall["breakdown"]
            lines.extend([
                "",
                "   Score Breakdown:",
                f"     - Completion: {breakdown.get('completion', 0):.1f}/40",
                f"     - Efficiency: {breakdown.get('efficiency', 0):.1f}/20",
                f"     - No Repetitions: {breakdown.get('repetition', 0):.1f}/15",
                f"     - Cleanup: {breakdown.get('cleanup', 0):.1f}/15",
                f"     - No Cycles: {breakdown.get('cycle', 0):.1f}/10",
            ])
        
        lines.extend([
            "",
            "=" * 70,
            f"Full logs available at: {self.run_dir}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def aggregate_behavior_metrics(game_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate behavior metrics across multiple games.
        
        Args:
            game_results: List of game results with behavior_metrics
            
        Returns:
            Aggregated metrics dictionary
        """
        if not game_results:
            return {}
        
        # Collect all metrics
        repeated_counts = []
        repetition_rates = []
        max_repeats = []
        
        total_opens = []
        correct_close_counts = []
        remaining_open_counts = []
        cleanup_rates = []
        
        cycles_detected = []
        cycle_rates = []
        wasted_steps = []
        
        overall_scores = []
        completion_scores = []
        efficiency_scores = []
        repetition_scores = []
        cleanup_scores = []
        cycle_scores = []
        grades = []
        
        for result in game_results:
            bm = result.get("behavior_metrics", {})
            
            rep = bm.get("repeated_steps", {})
            repeated_counts.append(rep.get("repeated_count", 0))
            repetition_rates.append(rep.get("repetition_rate", 0))
            max_repeats.append(rep.get("max_repeat_length", 0))
            
            clean = bm.get("cleanup", {})
            game_total_opens = clean.get("total_opens", clean.get("items_opened", 0))
            total_opens.append(game_total_opens)
            correct_close_counts.append(clean.get("correct_close_count", clean.get("items_closed", 0)))
            remaining_open_counts.append(clean.get("remaining_open_count", 0))
            # Only include cleanup rate if the game actually had opens
            if game_total_opens > 0:
                cleanup_rates.append(clean.get("cleanup_rate", 0.0))
            
            cyc = bm.get("cycles", {})
            cycles_detected.append(cyc.get("cycles_detected", 0))
            cycle_rates.append(cyc.get("cycle_rate", 0))
            wasted_steps.append(cyc.get("wasted_steps", 0))
            
            overall = bm.get("overall", {})
            overall_scores.append(overall.get("overall_score", 0))
            grades.append(overall.get("grade", "F"))
            
            breakdown = overall.get("breakdown", {})
            completion_scores.append(breakdown.get("completion_score", 0))
            efficiency_scores.append(breakdown.get("efficiency_score", 0))
            repetition_scores.append(breakdown.get("repetition_score", 0))
            cleanup_scores.append(breakdown.get("cleanup_score", 0))
            cycle_scores.append(breakdown.get("cycle_score", 0))
        
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        # Determine average grade
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        avg_grade_val = avg([grade_values.get(g, 0) for g in grades])
        avg_grade = "F"
        for grade, val in grade_values.items():
            if avg_grade_val >= val - 0.5:
                avg_grade = grade
                break
        
        return {
            "repeated_steps": {
                "avg_repeated_count": avg(repeated_counts),
                "avg_repetition_rate": avg(repetition_rates),
                "max_repeat_length": max(max_repeats) if max_repeats else 0,
            },
            "cleanup": {
                "avg_total_opens": avg(total_opens),
                "avg_correct_close_count": avg(correct_close_counts),
                "avg_remaining_open_count": avg(remaining_open_counts),
                "avg_cleanup_rate": avg(cleanup_rates),
            },
            "cycles": {
                "avg_cycles_detected": avg(cycles_detected),
                "avg_cycle_rate": avg(cycle_rates),
                "avg_wasted_steps": avg(wasted_steps),
            },
            "overall": {
                "avg_score": avg(overall_scores),
                "avg_grade": avg_grade,
                "breakdown": {
                    "completion": avg(completion_scores),
                    "efficiency": avg(efficiency_scores),
                    "repetition": avg(repetition_scores),
                    "cleanup": avg(cleanup_scores),
                    "cycle": avg(cycle_scores),
                }
            }
        }
