# Current Gate / Core Safety — new workflow

Users supply pre-split subsets; neither the workflow nor its optional LLM splits or reshuffles data. Initial submission has Train + Val + initial acceptance S0; each maintenance submission has new Train + new Current Gate. Existing Val/history are reused, and additional independent Val is optional. No extra research final-test dataset is required. A/B/C are successive batches with no fixed round count. See docs/DATA_PREPARATION.md for folder examples and split requirements.

Initial: independent Train and Val for modeling/calibration, plus the initial Core Safety S0 for independent evaluation. Initial training/evaluation completes, then the engineer adopts the exact weight+threshold contract. S0 is recorded once.

Maintenance r: new Train Dr + permitted historical training replay; reuse/add independent Val for calibration; independently evaluate the actual active parent and selected candidates on accumulated Sr-1 plus new Current Gate Gr. No evaluation or Val image enters training/mining/Fisher. ID, content SHA and optional building/source group are checked across roles, including prior and cancelled rounds.

After completed evaluation and explicit engineer adopt/Hold with reason, Sr=Sr-1+Gr. A draft, failed unevaluated task or cancellation cannot accumulate Gr. An evaluated HNM empty pool can Hold its exact parent. Source/checkpoint/data/threshold hashes bind the evidence. Historical regression is advisory; the engineer chooses. Rollback to a previously adopted version is a separate recorded operation and does not erase protected historical data.

No FS review or clustering is required. Three backbones and eight tools are engineering options, not a compulsory experimental matrix. Chat and auxiliary panels invoke the same transactional service. One human chat submission counts once; its automatic substeps do not add human operations.
