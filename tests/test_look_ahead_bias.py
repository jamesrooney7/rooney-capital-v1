#!/usr/bin/env python3
"""
Automated Look-Ahead Bias Scanner

Scans all strategy files for common look-ahead bias patterns.
This catches the most critical issues automatically.

Author: Rooney Capital
Date: 2025-01-22
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


class LookAheadBiasScanner:
    """Automated scanner for look-ahead bias in strategy code."""

    def __init__(self, strategy_dir: Path):
        self.strategy_dir = strategy_dir
        self.issues = []

    def scan_all_strategies(self) -> Dict[str, List[Dict]]:
        """Scan all strategy files for look-ahead bias."""
        results = {}

        strategy_files = list(self.strategy_dir.glob('*_bt.py'))

        print(f"Scanning {len(strategy_files)} strategy files...")
        print("="*80)

        for strategy_file in sorted(strategy_files):
            issues = self.scan_file(strategy_file)
            if issues:
                results[strategy_file.name] = issues

        return results

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single file for look-ahead bias patterns."""
        issues = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Pattern 1: Future bar access (most critical!)
        future_access_pattern = r'self\.data\.\w+\[([1-9]\d*)\]'

        # Pattern 2: Positive index on indicators
        indicator_future_pattern = r'self\.\w+\[([1-9]\d*)\]'

        # Pattern 3: Using next bar open in current bar decision
        next_open_pattern = r'self\.data\.open\[1\]'

        for i, line in enumerate(lines, 1):
            # Check for future bar access
            future_matches = re.findall(future_access_pattern, line)
            for match in future_matches:
                # Ignore comments
                if not line.strip().startswith('#'):
                    issues.append({
                        'line': i,
                        'code': line.strip(),
                        'severity': 'CRITICAL',
                        'type': 'FUTURE_BAR_ACCESS',
                        'description': f'Accessing future bar data with index [{match}]',
                        'fix': 'Use [0] for current bar or [-N] for historical bars'
                    })

            # Check for future indicator values
            if 'self.data' not in line:  # Already checked above
                indicator_matches = re.findall(indicator_future_pattern, line)
                for match in indicator_matches:
                    if not line.strip().startswith('#'):
                        issues.append({
                            'line': i,
                            'code': line.strip(),
                            'severity': 'HIGH',
                            'type': 'FUTURE_INDICATOR_ACCESS',
                            'description': f'Accessing future indicator value with index [{match}]',
                            'fix': 'Use [0] for current value or [-N] for historical values'
                        })

            # Check for specific risky patterns
            if 'cheat_on_close' in line.lower() and 'true' in line.lower():
                if not line.strip().startswith('#'):
                    issues.append({
                        'line': i,
                        'code': line.strip(),
                        'severity': 'MEDIUM',
                        'type': 'CHEAT_ON_CLOSE',
                        'description': 'Using cheat_on_close=True (unrealistic for live trading)',
                        'fix': 'Set to False for realistic backtesting or document justification'
                    })

        return issues

    def check_atr_usage(self, file_path: Path) -> List[Dict]:
        """Check ATR stop loss implementation for look-ahead bias."""
        issues = []

        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Look for ATR-based stops
        if 'atr' in content.lower():
            for i, line in enumerate(lines, 1):
                # Check if using future ATR for stops
                if 'stop' in line.lower() and 'self.atr[1]' in line:
                    issues.append({
                        'line': i,
                        'code': line.strip(),
                        'severity': 'CRITICAL',
                        'type': 'FUTURE_ATR_STOP',
                        'description': 'Using future ATR[1] for stop loss calculation',
                        'fix': 'Use self.atr[0] (current bar ATR)'
                    })

        return issues

    def generate_report(self, results: Dict[str, List[Dict]]) -> str:
        """Generate formatted report of findings."""
        report = []
        report.append("="*80)
        report.append("LOOK-AHEAD BIAS SCAN REPORT")
        report.append("="*80)
        report.append("")

        if not results:
            report.append("✅ NO CRITICAL ISSUES FOUND!")
            report.append("")
            report.append("All scanned files appear free of obvious look-ahead bias.")
            return "\n".join(report)

        # Count issues by severity
        critical_count = 0
        high_count = 0
        medium_count = 0

        for file_issues in results.values():
            for issue in file_issues:
                if issue['severity'] == 'CRITICAL':
                    critical_count += 1
                elif issue['severity'] == 'HIGH':
                    high_count += 1
                elif issue['severity'] == 'MEDIUM':
                    medium_count += 1

        report.append(f"Total files with issues: {len(results)}")
        report.append(f"CRITICAL issues: {critical_count}")
        report.append(f"HIGH issues: {high_count}")
        report.append(f"MEDIUM issues: {medium_count}")
        report.append("")

        # Detail each file
        for filename, issues in sorted(results.items()):
            report.append("="*80)
            report.append(f"FILE: {filename}")
            report.append("="*80)
            report.append("")

            for issue in issues:
                severity_marker = {
                    'CRITICAL': '🚨',
                    'HIGH': '⚠️',
                    'MEDIUM': '⚡'
                }.get(issue['severity'], '•')

                report.append(f"{severity_marker} {issue['severity']}: {issue['type']}")
                report.append(f"   Line {issue['line']}: {issue['description']}")
                report.append(f"   Code: {issue['code']}")
                report.append(f"   Fix: {issue['fix']}")
                report.append("")

        # Summary
        report.append("="*80)
        report.append("SUMMARY & RECOMMENDATIONS")
        report.append("="*80)
        report.append("")

        if critical_count > 0:
            report.append("🚨 CRITICAL ISSUES FOUND!")
            report.append("")
            report.append("Action Required:")
            report.append("1. Review all CRITICAL issues immediately")
            report.append("2. Fix future bar access ([1] or positive indices)")
            report.append("3. Verify ATR stops use current bar ATR")
            report.append("4. Re-run scanner after fixes")
            report.append("")

        if high_count > 0:
            report.append("⚠️  HIGH PRIORITY ISSUES FOUND")
            report.append("")
            report.append("Action Recommended:")
            report.append("1. Review all HIGH issues")
            report.append("2. Verify indicator calculations")
            report.append("3. Check for unintended future data usage")
            report.append("")

        if medium_count > 0:
            report.append("⚡ MEDIUM PRIORITY ISSUES FOUND")
            report.append("")
            report.append("Action Suggested:")
            report.append("1. Review cheat_on_close usage")
            report.append("2. Document justifications")
            report.append("3. Consider realistic settings for live trading")
            report.append("")

        return "\n".join(report)


def main():
    print("="*80)
    print("AUTOMATED LOOK-AHEAD BIAS SCANNER")
    print("="*80)
    print()
    print("Scanning all Backtrader strategy implementations...")
    print()

    strategy_dir = Path('src/strategy/strategy_factory')

    if not strategy_dir.exists():
        print(f"❌ Strategy directory not found: {strategy_dir}")
        return 1

    scanner = LookAheadBiasScanner(strategy_dir)
    results = scanner.scan_all_strategies()

    print()
    report = scanner.generate_report(results)
    print(report)

    # Save report
    report_path = Path('docs/look_ahead_bias_scan_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    print()
    print(f"📄 Full report saved to: {report_path}")
    print()

    # Return exit code based on findings
    if results:
        critical_issues = sum(
            1 for issues in results.values()
            for issue in issues
            if issue['severity'] == 'CRITICAL'
        )
        if critical_issues > 0:
            print("❌ CRITICAL ISSUES FOUND - Fix before deployment!")
            return 1
        else:
            print("⚠️  Non-critical issues found - Review recommended")
            return 0
    else:
        print("✅ No obvious look-ahead bias detected")
        return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
