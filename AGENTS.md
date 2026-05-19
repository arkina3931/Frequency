<claude-mem-context>
# Memory Context

# [Frequency] recent context, 2026-05-19 9:08pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 14 obs (4,274t read) | 71,968t work | 94% savings

### May 19, 2026
454 8:51p 🔵 MMRec framework codebase evaluation requested
455 " 🔵 Frequency framework is an MMRec-family multimodal recommendation system
456 8:52p 🔵 Full dataloader.py structure revealed for evaluation
457 " 🔵 Multiple improvement opportunities identified in MMRec/Frequency framework
458 8:53p 🔵 Frequency framework has no dependency specification files
459 8:54p 🔵 Frequency framework Python environment has no ML dependencies installed
460 " 🔴 LayerGCN model imports from non-existent package path
461 8:56p ⚖️ User prioritized training speed optimization for Frequency framework
462 8:57p ⚖️ User chose conservative speed optimization approach
**463** 9:00p ✅ **Duplicate config directory src/config/ removed**
The session removed the duplicate src/config/ directory containing overall.yaml and mg.yaml. Earlier analysis confirmed these files were identical to those in src/configs/, and the Configurator class only loads configuration from the src/configs/ path. The src/config/ directory was also incomplete, lacking the dataset/ and model/ subdirectories that src/configs/ maintains.

**464** 9:02p 🟣 **Conservative speed optimization applied to TrainDataLoader and EvalDataLoader**
Applied a comprehensive conservative speed optimization to src/utils/dataloader.py. The core changes eliminate DataFrame slicing overhead in the hot batch iteration loop by pre-caching uid and iid columns as numpy arrays. Tensor construction is streamlined by using torch.as_tensor with direct device placement (avoiding the old pattern of CPU tensor creation → .type() → .to(device) triple-hop). The EvalDataLoader's batch assembly was reworked to use pre-computed cumulative offset arrays, replacing O(n) sum() calls with O(1) index lookups. Random sampling was micro-optimized by using random.choice() instead of random.sample(x, 1)[0], and boundary guards were added throughout. These changes preserve training trajectory and metric comparability as required by the user's "稳妥提速" constraint.

**465** 9:03p 🟣 **Conditional test evaluation and runtime profiling added to Trainer**
The Trainer.fit() method was extended with two opt-in config options. test_eval_mode replaces the unconditional test evaluation every epoch with three modes — the default 'on_valid_update' runs the full test forward pass only when validation score improves, which saves significant time since most epochs don't trigger improvement in typical training runs. profile_runtime adds detailed per-epoch timing breakdowns for diagnosing bottlenecks. Both changes preserve training trajectory and metric comparability: test_eval_mode only skips test evaluations that would not affect the tracked best_test_upon_valid, and profile_runtime only adds logging.

**467** " ✅ **New config options added to overall.yaml**
The overall.yaml default configuration was updated with three new keys supporting the speed optimization work. test_eval_mode defaults to 'on_valid_update' (test only on validation improvement), profile_runtime defaults to False (opt-in profiling), and use_mm_adj_cache defaults to True (SSR adjacency matrix caching enabled).

**468** 9:04p 🔵 **README.md was UTF-16 encoded, causing patch tool failure**
During the README update, the session discovered that README.md was saved in UTF-16 LE encoding (detected via Format-Hex showing the FF FE BOM). The original content was only "Frequency Framework" but in double-byte encoding. The file was rewritten as UTF-8 and the dependency installation instructions were successfully added.


Access 72k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>