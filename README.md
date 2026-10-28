# voicedecide

Training-free decisions on audio: **call routing** and **language identification** with
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). Audio in, a probability for every
option out, plus an `act` / `confirm` / `escalate` action your pipeline can branch on.

No fine-tuning and no text generation. A clip is embedded once, then compared with each option:

```
audio → EmbeddingGemma 2 (frozen) → cosine vs. option vectors → ÷ τ → softmax → decision
```

- **Zero-shot (default):** describe the options in words.
- **Few-shot prototypes (recommended):** give ~10 labeled clips per option, build once, reuse.

## Background

voicedecide grew out of an experiment that tested EmbeddingGemma 2 as a training-free voice decision
layer on call routing, language identification and emotion recognition. The notebook with the
original setup and evaluation is on Kaggle:
[EmbeddingGemma 2 as a training-free voice decision layer](https://www.kaggle.com/code/palsnehal004/embeddinggemma-2-as-a-training-free-voice-jev?scriptVersionId=363227677).
Routing and language ID worked well there, so the library supports those two tasks. Emotion from tone
alone did not, so it is not included.

## Install

```bash
pip install voicedecide
voicedecide download        # optional: fetch the model ahead of time (~2.5 GB, cached)
```

The model weights are not part of the package. They are downloaded from Hugging Face on first use
and cached in `~/.cache/huggingface` (`HF_HOME` moves it; `HF_HUB_OFFLINE=1` runs without network
once cached). Python 3.11+. A GPU is recommended for live use; CPU works for batch jobs.

## Quick start: zero-shot routing (default)

```python
from voicedecide import Decider, Choice

d = Decider()

route = Choice.from_descriptions({
    "card_issues": ["A caller reporting a lost or stolen card",
                    "Someone whose bank card is blocked or damaged"],
    "balance":     ["A caller asking how much money is in their account"],
    "pay_bill":    ["A caller wanting to pay a bill"],
})

res = d.decide("call.wav", {"route": route})
a = res["route"]
print(a.choice, a.probability, a.action)
```

Write descriptions as full phrases ("A caller reporting a lost card"), not bare labels ("lost card").
Several phrasings per option are averaged.

## Recommended: build prototypes once (few-shot)

```python
from glob import glob

route = Choice.from_examples({          # ~10 of YOUR calls per option, from different speakers
    "card_issues": glob("labeled/card_issues/*.wav"),
    "balance":     glob("labeled/balance/*.wav"),
    "pay_bill":    glob("labeled/pay_bill/*.wav"),
}, encoder=d)

report = route.fit(val_files, val_labels, encoder=d)   # different clips: picks τ and thresholds
print(report)                                          # accuracy, macro-F1, ECE, action shares
route.save("packs/my-routing")                         # vectors + settings only, no audio
```

Nothing is trained: each option's clip embeddings are averaged into one prototype. Use clips from
your own audio (same phone line, noise, accents). Prototypes from studio audio tested on phone calls
is the most common way accuracy drops.

## In a pipeline

```python
d = Decider()
route = Choice.from_pack("packs/my-routing")
lang = Choice.from_pack("language-id")

def handle_call(audio_bytes):
    res = d.decide(audio_bytes, {"route": route, "lang": lang})   # one embedding, two answers
    r = res["route"]
    if r.action == "act":
        return transfer(r.choice, language=res["lang"].choice)
    if r.action == "confirm":
        return ask(f"Did you mean {r.choice.replace('_', ' ')}?")
    return to_human_agent()
```

`audio` can be a file path, encoded bytes, a file object, `(array, sample_rate)`, or a Hugging Face
audio dict. Audio is converted to mono 16 kHz. `max_seconds=5` decides on the start of the clip only.

### Result

```python
a = res["route"]
a.choice          # "card_issues" (None if the audio matches no option, see out_of_set)
a.probabilities   # {"card_issues": ..., "balance": ..., "pay_bill": ...}  sums to 1
a.confidence      # 0-1: 1 - normalized entropy (how concentrated the probabilities are)
a.margin          # top-1 minus top-2 probability
a.action          # "act" | "confirm" | "escalate"
a.out_of_set      # True when the best match is below the question's calibrated floor
res.latency_ms    # {"decode", "embed", "decide", "total"}
res.to_json()
```

`act` and `confirm` thresholds default to 0.80 / 0.50 and are replaced by `fit()`, which picks the
lowest probability whose decisions reach 95 % (act) and 80 % (confirm) precision on your held-out clips.
`Decider(strict=True)` raises `OutOfSetError` instead of returning out-of-set answers.

## Language identification

```python
lang = Choice.from_pack("language-id")          # tested languages only
lang = Choice.from_pack("language-id", languages=["hindi", "ta", "gu_in"])
```

**Tested languages:** English, Hindi, Bengali, Tamil, Telugu, Marathi. These load by default.

The pack also contains these languages, which **have not been tested**: Urdu, Gujarati, Kannada,
Malayalam, Odia, Punjabi, Assamese, Nepali, Sindhi, Mandarin Chinese, Cantonese, Spanish, Portuguese,
French, Italian, German, Dutch, Russian, Ukrainian, Polish, Turkish, Arabic, Persian, Japanese, Korean,
Vietnamese, Thai, Burmese, Indonesian, Malay, Filipino, Swahili, Hausa, Amharic.

Selecting an untested language works but warns once:

```
UntestedLanguageWarning: 'gu_in' (Gujarati) has not been tested.
  Tested languages: Bengali, English, Hindi, Marathi, Tamil, Telugu.
  Results for untested languages may be unreliable. Check them on your own labeled clips:
  voicedecide eval --pack language-id --languages gu_in your_clips.csv
```

The built-in pack has no out-of-set floor: one calibrated on FLEURS read speech also flags correctly
identified phone calls. To detect speech in languages outside the pack, run `lang.fit(clips, labels)`
on labeled clips from your own pipeline; that sets a floor matched to your audio.

A language that is not in the pack raises `UnknownLanguageError` with a suggestion. You can add your
own with `lang.add_examples({"xx": clips})` (marked untested until you run `fit` on your clips).
Closely related languages (for example Hindi and Urdu) are expected to be confused more often.

`voicedecide languages` prints every language and its status.

## Command line

```bash
# folder layout: labeled/<option>/*.wav
voicedecide build-pack --examples labeled/ --val val/ --out packs/my-routing
voicedecide run --pack route=packs/my-routing --pack language-id calls/*.wav > decisions.jsonl
voicedecide eval --pack packs/my-routing test.csv          # CSV columns: path,label
voicedecide languages
```

## Options worth knowing

| | |
|---|---|
| `Choice.from_descriptions(..., centering=True)` + `q.adapt(unlabeled_clips)` | subtract each modality's mean (no labels needed) |
| `q.add_examples({...})`, `q.remove_label(...)`, `q.subset([...])` | edit a question without rebuilding |
| `Decider(revision="<commit>")` | pin the model version; packs record the revision they were built with |
| `Decider(device="cpu", dtype=torch.float32)` | float16 is refused: this model returns NaN or degraded vectors in fp16 |
| `d.decide_batch(files, questions, batch_size=8)` | batched embedding for offline jobs |

## Limits

- Audio is cut at 320 s (model context).
- Benchmarks use public datasets (FLEURS read speech, MINDS-14 banking calls). Real calls with
  8 kHz audio, noise or code-mixed speech can score lower: run `voicedecide eval` on your own labeled
  clips before going live.
- Out-of-set detection is approximate; a closely related language or intent can still be accepted.
  The floor set by `fit()` only holds for audio like the calibration clips (same channel and style).
- A small or very easy calibration set pushes τ to the edge of its grid and makes probabilities
  over-confident; `fit()` notes this in its report. Use 20+ held-out clips per option when you can.

## Reproducing the numbers

- `benchmarks/build_language_pack.py` builds the `language-id` pack from FLEURS (validation split for
  prototypes and calibration, test split for the benchmark) and records the benchmark in `pack.json`.
- `examples/banking-routing/build_and_eval.py` builds a routing pack from MINDS-14 en-GB/en-AU and
  evaluates it on en-US, next to zero-shot text anchors.

## Contributing

voicedecide is maintained in spare time, so reviews may take a few weeks. Please open an issue before
a pull request; bug fixes with tests are the most likely to be merged. Packs are shared on the
Hugging Face Hub, not in this repository. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Datasets and credits

The library ships no audio. These datasets were used in the original experiment, the benchmarks and the
built-in pack:

| Dataset | Used for | License | Credit |
|---|---|---|---|
| [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) (PolyAI) | call-routing experiment, `examples/banking-routing` | CC BY 4.0 | Gerz et al., *Multilingual and Cross-Lingual Intent Detection from Spoken Data*, 2021, [arXiv:2104.08524](https://arxiv.org/abs/2104.08524) |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) | language-ID experiment, built-in `language-id` pack vectors | CC BY 4.0 | Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, 2022, [arXiv:2205.12446](https://arxiv.org/abs/2205.12446) |
| [RAVDESS](https://zenodo.org/records/1188976) ([HF mirror](https://huggingface.co/datasets/xbgoose/ravdess)) | emotion experiment in the Kaggle notebook only; nothing in this library is derived from it | CC BY-NC-SA 4.0 | Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLoS ONE 13(5): e0196391, 2018, [doi:10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391) |

Model: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) by Google DeepMind
(Apache 2.0), *EmbeddingGemma 2: Unified and Efficient Multimodal Representations*, 2026,
[arXiv:2610.37364](https://arxiv.org/abs/2610.37364).

## License

Apache 2.0. EmbeddingGemma 2 is Google DeepMind's model (Apache 2.0), downloaded at runtime. Built-in
pack vectors are derived from FLEURS (CC BY 4.0). See `NOTICE`.

Created by [Snehal Pal](https://snehalpal.vercel.app/).
