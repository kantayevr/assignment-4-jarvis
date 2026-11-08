Assignment 4 GitHub Releases holdout v1

The dataset contains 34 test questions. Each question uses one of the 34 eligible
repositories from the 40 candidates in Appendix A. The frozen and updated cutoffs
are 1 January 2025 and 1 June 2025 at 00:00 UTC, respectively.

questions.json contains the test prompts. answer_key.json is for scoring.
kb_frozen.json and kb_updated.json contain the two release snapshots.
collection_ledger.json records all 40 candidates and explains the six exclusions.
dataset_manifest.json records counts, SHA-256 digests and the source archive digests.

release_metadata contains the saved factual fields from the paginated GitHub REST
API release lists. Each archived page records the requested URL, retrieval time
and SHA-256 digest of the retained metadata. The historical snapshots reflect
the releases returned at collection time.

To verify the files offline after extracting the archive, run the following
from the extracted directory with Python 3.10 or later.

    python collect_assignment4.py --output . --verify
