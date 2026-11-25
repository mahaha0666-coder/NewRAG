# Experiment summaries

| File | Evaluation |
| --- | --- |
| [comparative-baselines.csv](comparative-baselines.csv) | Evidence-first MechRAG and eight DeepSeek Flash configurations on 1,185 public MechVQA questions and 26 manually verified fields from the local drawing set |
| [mechvqa.md](mechvqa.md) | Overall, capability and difficulty scores for evidence-first MechRAG on the public MechVQA test set |
| [cad-deepseek-pilot.csv](cad-deepseek-pilot.csv) | Document-local retrieval pilot using DeepSeek Flash across DeepCAD, TriView2CAD and FCM |
| [cad-qwen-pilot.csv](cad-qwen-pilot.csv) | Direct image answering, OCR + BM25 and OCR + BGE-M3 using local Qwen3-VL-2B-Instruct Q4 |
| [cad-qwen-deepcad.csv](cad-qwen-deepcad.csv) and [cad-qwen-fcm.csv](cad-qwen-fcm.csv) | Both local Qwen OCR baselines on 8,052 derived DeepCAD test sketches and 53 labelled FCM images |

The MechVQA run uses public benchmark revision `c8d61513aef05b4f5d117bb0c6d1a055451d6f33`, with DeepSeek Flash for generation and automated judging. It evaluates question answering with the paired drawing supplied for each question. The eight baselines use the same visual model with two view configurations and four prompting strategies.

The TriView2CAD pilot uses the same 100 synthetic drawings for the three local Qwen configurations. Each question scores three counts against CAD metadata. The [manuscript table](../elsevier_evidence_rag/main.tex#L139) reports preliminary baseline results. Full GeoEvidence-RAG results and component ablations are pending.
