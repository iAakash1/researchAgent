Recorded Crossref REST API responses used by the integrity-registry tests.

Recorded on 2026-10-10 from `https://api.crossref.org/works/{doi}` and
`https://api.crossref.org/works?filter=updates:{doi}`, then trimmed to the fields the
adapter reads (`DOI`, `type`, `title`, `updated-by`, `update-to`). Values are unedited.

| File | DOI | What it shows |
|---|---|---|
| `work_retracted_retraction_watch.json` | 10.1021/am300292v | `updated-by` retraction, source `retraction-watch`, `record-id` |
| `work_retracted_two_sources.json` | 10.1177/1758835920922055 | the same retraction from `retraction-watch` and `publisher` |
| `notice_retraction.json` | 10.1177/17588359231172420 | the notice side: `update-to` pointing at the retracted work |
| `work_expression_of_concern.json` | 10.1016/j.lungcan.2003.10.006 | `expression_of_concern`, two dated entries |
| `work_corrected.json` | 10.1088/1361-6595/aaebdb | `correction` |
| `work_withdrawn_self_notice.json` | 10.1016/j.adengl.2021.11.027 | `withdrawal` whose notice is the work itself |
| `work_current.json` | 10.1038/nature14539 | no updates |
| `updates_retracted_two_sources.json` | filter `updates:10.1177/1758835920922055` | notice listing |
| `updates_current.json` | filter `updates:10.1038/nature14539` | empty listing |

Synthetic cases (lookup failures, unmapped update types) are built in the tests with
Crossref's test prefix `10.5555`, which is never a real registered work.
