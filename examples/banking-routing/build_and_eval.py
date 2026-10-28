"""Build a call-routing pack from MINDS-14 and compare it with zero-shot text anchors.

Prototypes + calibration come from British and Australian English calls (en-GB, en-AU);
evaluation uses American English calls (en-US), so the number reflects a change of accent
and speakers, like a real deployment would.

    pip install voicedecide pyarrow pandas
    python examples/banking-routing/build_and_eval.py --out packs/banking-routing
"""

from __future__ import annotations

import argparse
import random

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from voicedecide import Choice, Decider

INTENTS = ["abroad", "address", "app_error", "atm_limit", "balance", "business_loan", "card_issues", "cash_deposit",
           "direct_debit", "freeze", "high_value_payment", "joint_account", "latest_transactions", "pay_bill"]

DESCRIPTIONS = {
    "abroad": ["A caller asking whether their card will work abroad", "A customer telling the bank about a trip overseas"],
    "address": ["A caller wanting to change the address on their account", "Someone who moved and needs to update their address"],
    "app_error": ["A caller reporting that the banking app is not working", "Someone who cannot log in to the app"],
    "atm_limit": ["A caller asking about the daily ATM withdrawal limit", "Someone wanting to take out more cash at a cash machine"],
    "balance": ["A caller asking how much money is in their account", "Someone wanting to check their balance"],
    "business_loan": ["A caller asking about a loan for their business", "A business owner asking about financing"],
    "card_issues": ["A caller reporting a lost or stolen card", "Someone whose bank card is blocked or damaged"],
    "cash_deposit": ["A caller asking how to deposit cash", "Someone wanting to pay money in at a branch"],
    "direct_debit": ["A caller wanting to set up or cancel a direct debit", "Someone asking about a recurring payment"],
    "freeze": ["A caller asking to freeze their account", "Someone wanting to lock their card or account"],
    "high_value_payment": ["A caller wanting to make a very large transfer", "Someone asking about the limit for big payments"],
    "joint_account": ["A caller wanting to open a joint account", "Someone asking to add their partner to an account"],
    "latest_transactions": ["A caller asking about their recent transactions", "Someone asking about a payment they don't recognise"],
    "pay_bill": ["A caller wanting to pay a bill", "Someone asking how to pay their phone or utility bill"],
}


def minds(locale: str):
    path = hf_hub_download("PolyAI/minds14", f"{locale}/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(path, columns=["audio", "intent_class"]).to_pandas()
    return [(a, INTENTS[i]) for a, i in zip(t.audio, t.intent_class)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="packs/banking-routing")
    ap.add_argument("--per-intent", type=int, default=10)
    args = ap.parse_args()
    random.seed(42)

    train = minds("en-GB") + minds("en-AU")
    random.shuffle(train)
    test = minds("en-US")
    examples, calib = {i: [] for i in INTENTS}, []
    for a, label in train:  # first N clips per intent build prototypes, the rest calibrate
        if len(examples[label]) < args.per_intent:
            examples[label].append(a)
        else:
            calib.append((a, label))
    d = Decider()

    zero_shot = Choice.from_descriptions(DESCRIPTIONS, name="banking-routing-zero-shot")
    print("zero-shot text anchors on en-US:\n", zero_shot.evaluate([a for a, _ in test], [l for _, l in test], encoder=d))

    q = Choice.from_examples(examples, encoder=d, name="banking-routing")
    print("\ncalibration on en-GB/en-AU:\n", q.fit([a for a, _ in calib], [l for _, l in calib], encoder=d))
    print("\nprototypes on en-US:\n", q.evaluate([a for a, _ in test], [l for _, l in test], encoder=d))
    print("\nsaved", q.save(args.out))


if __name__ == "__main__":
    main()
