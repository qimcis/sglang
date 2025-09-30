import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tiktoken

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    dump_bench_raw_result,
    select_sglang_backend,
)

choices = ["A", "B", "C", "D"]

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def get_few_shot_examples_list(train_df, subject, k=-1):
    if k == -1:
        k = train_df.shape[0]
        
    examples = []
    for i in range(k):
        question = train_df.iloc[i, 0]
        k_options = train_df.shape[1] - 2
        for j in range(k_options):
            question += "\n{}. {}".format(choices[j], train_df.iloc[i, j + 1])
        question += "\nAnswer:"

        answer = " {}\n\n".format(train_df.iloc[i, k_options + 1])
        
        examples.append({
            "question": question,
            "answer": answer
        })
    return examples

def main(args):
    set_default_backend(select_sglang_backend(args))
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # Build prompts
    all_questions = []
    labels = []
    num_questions = []

    for subject in subjects[: args.nsub]:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        num_questions.append(test_df.shape[0])

        k = args.ntrain
        few_shot_examples = get_few_shot_examples_list(dev_df, subject, k)
        
        # while len(tokenizer.encode(few_shot_examples)) > 1536:
        #    k -= 1
        #    few_shot_examples = get_few_shot_examples_list(dev_df, subject, k)

        for i in range(test_df.shape[0]):
            question = format_example(test_df, i, include_answer=False)

            all_questions.append(
                {
                    "few_shot_examples": few_shot_examples,
                    "question": question,
                }
            )

            label = test_df.iloc[i, test_df.shape[1] - 1]
            labels.append(label)

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_mmlu(s, few_shot_examples, question):
        for example in few_shot_examples:
            s += sgl.user(example["question"])
            s += sgl.assistant(example["answer"])

        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer"))
        
    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    # Run
    tic = time.perf_counter()
    states = few_shot_mmlu.run_batch(
        all_questions,
        temperature=0,
        max_new_tokens=1,
        num_threads=args.parallel,
        progress_bar=True,
    )
    preds = [
        s["answer"].strip()[0] if len(s["answer"].strip()) > 0 else "" for s in states
    ]
    latency = time.perf_counter() - tic

    # Compute accuracy
    cors = [pred == label for pred, label in zip(preds, labels)]

    pt = 0
    for subject, num_qs in zip(subjects[: args.nsub], num_questions):
        print(
            f"subject: {subject}, #q:{num_qs}, acc: {np.mean(cors[pt: pt + num_qs]):.3f}"
        )
        pt += num_qs
    assert pt == len(cors)
    weighted_acc = np.mean(cors)

    dump_bench_raw_result(
        path=args.raw_result_file,
        states=states,
        preds=preds,
        labels=labels,
    )

    # Print results
    print("Total latency: {:.3f}".format(latency))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "mmlu",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(weighted_acc, 3),
            "num_requests": len(all_questions),
            "other": {
                "nsub": args.nsub,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--nsub", type=int, default=60)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
