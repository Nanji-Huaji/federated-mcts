"""Render the archive-based control without merging it with the 30-puzzle cohort."""
from decimal import Decimal


def paragraph(s):
    n=s['tasks']; k=s['complete']['solved_in_first_k']; prs=s['primary_paired_full_tree_vs_complete_k']['10']
    assert s['pairing']['matched_slots']==10*n
    totals=s['usage']['totals_by_arm']; b,t=totals['complete_k10'],totals['tree_full']
    tree=prs['tree_solved']; assert prs['complete_solved']==k['10'] and tree-prs['complete_solved']==prs['tree_only']-prs['complete_only']
    ratio=Decimal(t['cost'])/Decimal(b['cost']); ci=prs['paired_task_bootstrap95_pp']
    return (r'\subsection{Does the tree beat repeated complete answers?}'+'\n'+
        f'On a different cohort of {n} archived puzzles, we compared two DeepSeek-V3 pipelines: repeated complete answers with an external terminal verifier, and a full all-cloud tree. The original recipe and data snapshot agree with the inputs at every tree layer; the responses are tied to their runs, but individual request prompts and independent seeds are unavailable. We checked the final expression of each complete answer and the complete trace of each retained tree output using the same four-number, nonnegative-integer and exact-division rules. '
        f'The first one, three and ten complete answers solved {k["1"]}, {k["3"]} and {k["10"]} puzzles. The full tree solved {tree} (tree minus ten answers: {prs["tree_minus_complete_pp"]:+.2f} points, paired 95\\% interval $[{ci[0]:+.2f}, {ci[1]:+.2f}]$). '
        f'The ten-answer baseline used {b["total_calls"]:,} calls and cost an estimated CNY~{Decimal(b["cost"]):.2f}; the tree used {t["total_calls"]:,} calls at CNY~{Decimal(t["cost"]):.2f} ({ratio:.2f} times as much). These are recorded budgets, not equal-compute runs; sampling was not stopped early when an answer passed. The comparison motivates retaining complete-answer sampling as a baseline for any budgeted search policy.')
