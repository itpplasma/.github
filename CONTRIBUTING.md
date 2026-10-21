# Contributing

Never commit non-public machine, experiment or collaborator data as fixtures
(including ITER/TC24/IMAS, AUG shots and JT-60SA inputs). Use synthetic or
published fixtures with provenance. Keep campaign outputs outside Git with
checksums and a recorded location.

Data fixtures belong only in explicitly allowlisted fixture directories.
Record each path glob and its reason in the tracked `.repo-hygiene-allow`
file, separated by a TAB; secret detection has no exceptions. See
[repository hygiene](REPO_HYGIENE.md) for limits and the reusable CI gate.

Squash standalone PRs to one commit. Squash each layer of a stack separately;
after a lower layer merges, rebase the next layer with
`git rebase --onto origin/main OLD_LOWER_HEAD`, then retarget its PR to main.
Preserve each layer's independent changes and coordinate before rewriting
shared branches.

Heavy CI runs only in public repositories. Private repositories validate on
clusters; record job IDs, revisions, commands and results in the PR.
